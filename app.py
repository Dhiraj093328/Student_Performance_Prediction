import base64
import math
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "student_performance_model.pkl"
PHOTO_PATH = APP_DIR / "assets" / "profile.jpg"
DEVELOPER = "Dhiraj Waghmare"
INITIALS = "".join(w[0] for w in DEVELOPER.split()[:2]).upper()

# Profile photo embedded in the code so it always shows, even if the assets folder is missing.
_EMBEDDED_PHOTO_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/6xaJSlACEQAAAAEAABZ/anVtYgAAAB5qdW1kYzJwYQARABCAAACqADibcQNjMnBhAAAAFllq"
    "dW1iAAAAR2p1bWRjMm1hABEAEIAAAKoAOJtxA3VybjpjMnBhOjYyMGRlYTBjLWVjNTAtNDZiMy1iYjI5LTc4NjQ0YWZjOTE4NgAA"
    "AAOUanVtYgAAAClqdW1kYzJhcwARABCAAACqADibcQNjMnBhLmFzc2VydGlvbnMAAAAAuWp1bWIAAABEanVtZGNib3IAEQAQgAAA"
    "qgA4m3ETYzJwYS5pbmdyZWRpZW50LnYzAAAAABhjMnNoMpoDQZR36xhGsPwNmQ2fjQAAAG1jYm9yo2lkYzpmb3JtYXRqaW1hZ2Uv"
    "anBlZ2ppbnN0YW5jZUlEeCx4bXA6aWlkOmZiMjIyNDJlLTlhZDctNDEzNi05NDNhLTA0NWM4MDBmMzE5ZWxyZWxhdGlvbnNoaXBo"
    "cGFyZW50T2YAAAHianVtYgAAAEFqdW1kY2JvcgARABCAAACqADibcRNjMnBhLmFjdGlvbnMudjIAAAAAGGMyc2jrFDLUvItskudk"
    "0vnWHZv7AAABmWNib3KiZ2FjdGlvbnOComZhY3Rpb25rYzJwYS5vcGVuZWRqcGFyYW1ldGVyc6FraW5ncmVkaWVudHOBomN1cmx4"
    "LXNlbGYjanVtYmY9YzJwYS5hc3NlcnRpb25zL2MycGEuaW5ncmVkaWVudC52M2RoYXNoWCCIpeC1XvMRNKOcM6qvGJTLHNfRAX3b"
    "eWxmw9s30l5azaRmYWN0aW9ueB1jb20uYW50aHJvcGljLmNsYXVkZS5wcm92aWRlZGpwYXJhbWV0ZXJzoXgfY29tLmFudGhyb3Bp"
    "Yy5vcmlnaW4tY29uZmlkZW5jZWd1bmtub3dua2Rlc2NyaXB0aW9ueGZDbGF1ZGUgcHJvdmlkZWQgdGhpcyBmaWxlIGF0IHRoZSBy"
    "ZXF1ZXN0IG9mIGEgdXNlciBhbmQgbWF5IGhhdmUgY3JlYXRlZCBvciBtb2RpZmllZCB0aGUgZmlsZSBjb250ZW50cy5tc29mdHdh"
    "cmVBZ2VudKFkbmFtZWZDbGF1ZGVyYWxsQWN0aW9uc0luY2x1ZGVk9QAAAMhqdW1iAAAAQGp1bWRjYm9yABEAEIAAAKoAOJtxE2My"
    "cGEuaGFzaC5kYXRhAAAAABhjMnNoJu0D/4jdWao6wa93R6/fcwAAAIBjYm9ypWNhbGdmc2hhMjU2Y3BhZE4AAAAAAAAAAAAAAAAA"
    "AGRoYXNoWCDsGlBXhVQWYO6Fxhg0xfi7PqYM3vscgZInfqIhltdDt2RuYW1lbmp1bWJmIG1hbmlmZXN0amV4Y2x1c2lvbnOBomVz"
    "dGFydBRmbGVuZ3RoGRaLAAACPmp1bWIAAAAnanVtZGMyY2wAEQAQgAAAqgA4m3EDYzJwYS5jbGFpbS52MgAAAAIPY2JvcqVjYWxn"
    "ZnNoYTI1NmlzaWduYXR1cmV4TXNlbGYjanVtYmY9L2MycGEvdXJuOmMycGE6NjIwZGVhMGMtZWM1MC00NmIzLWJiMjktNzg2NDRh"
    "ZmM5MTg2L2MycGEuc2lnbmF0dXJlamluc3RhbmNlSUR4LHhtcDppaWQ6Zjg3OWZiMWYtOWZlMS00NTI5LWI2MmMtYjQ0YWExYmFm"
    "MjkzcmNyZWF0ZWRfYXNzZXJ0aW9uc4OiY3VybHgtc2VsZiNqdW1iZj1jMnBhLmFzc2VydGlvbnMvYzJwYS5pbmdyZWRpZW50LnYz"
    "ZGhhc2hYIIil4LVe8xE0o5wzqq8YlMsc19EBfdt5bGbD2zfSXlrNomN1cmx4KnNlbGYjanVtYmY9YzJwYS5hc3NlcnRpb25zL2My"
    "cGEuYWN0aW9ucy52MmRoYXNoWCB6Z0KYH1bX4voNdk9sgx6lOF4S0x5zC6onWd05sgLKfqJjdXJseClzZWxmI2p1bWJmPWMycGEu"
    "YXNzZXJ0aW9ucy9jMnBhLmhhc2guZGF0YWRoYXNoWCCOLbtPV6NYJHdWnDPYLDe9w7yRHt1UEPufssyfN2xCwXRjbGFpbV9nZW5l"
    "cmF0b3JfaW5mb6NkbmFtZW9BbnRocm9waWMgRmlsZXNndmVyc2lvbmUxLjAuMGtzcGVjVmVyc2lvbmUyLjQuMAAAEDhqdW1iAAAA"
    "KGp1bWRjMmNzABEAEIAAAKoAOJtxA2MycGEuc2lnbmF0dXJlAAAAEAhjYm9y0oRZAhKiASYYIVkCCjCCAgYwggGNoAMCAQICFEDl"
    "oAruwjnQvriD+gZCBT1nVRMAMAoGCCqGSM49BAMDMEkxFzAVBgNVBAoTDkFudGhyb3BpYywgUEJDMS4wLAYDVQQDEyVBbnRocm9w"
    "aWMgQ29udGVudCBDcmVkZW50aWFscyBSb290IENBMB4XDTI2MDgwNzE4NDM1NloXDTI4MDgwNjE5NDM1NlowRDEXMBUGA1UEChMO"
    "QW50aHJvcGljLCBQQkMxKTAnBgNVBAMTIEFudGhyb3BpYyBDbGF1ZGUgQ29udGVudCBTaWduaW5nMFkwEwYHKoZIzj0CAQYIKoZI"
    "zj0DAQcDQgAEmHoKa8tQGAUU1TS9QqU5W0Tp2N3XsvlK7BfQt6YWKwEzd2R3/dzKPEUDdCjlLjp9fT+KFjRVnuZ9v0oXvTe3k6NY"
    "MFYwDgYDVR0PAQH/BAQDAgeAMBUGA1UdJQQOMAwGCisGAQQBg+heAgEwDAYDVR0TAQH/BAIwADAfBgNVHSMEGDAWgBTOUeIEgU5k"
    "WyP448TPmj6cwddcwjAKBggqhkjOPQQDAwNnADBkAjAxcx0UngF60stVjs5G4T2eiptsBk5mf9oCtfJPAUBl8qs/PEXa8+gk1/X5"
    "QJ2DVcYCMHBfXN31YapiSqYvlIWrDVDJKOvXMl+kkz37Wt0PBI8sw486Mq6JeOhT+lRR4b1HCaFjcGFkWQ2eAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9lhA3M5SrTHKxLFdiRLzpLl8JdKWFuQ4mEwH469QeTP+lsrsaK8dbXSQMmifrI/9XZun"
    "EI9sDRRdGBudjiMKC6o3lv/bAEMABAMDAwMCBAMDAwQEBAUGCgYGBQUGDAgJBwoODA8ODgwNDQ8RFhMPEBURDQ0TGhMVFxgZGRkP"
    "EhsdGxgdFhgZGP/bAEMBBAQEBgUGCwYGCxgQDRAYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgY"
    "GBgYGP/AABEIAZABkAMBIgACEQEDEQH/xAAdAAABBAMBAQAAAAAAAAAAAAAAAQIDBAUGBwgJ/8QAPxAAAQMDAwIFAgQFAgUCBwAA"
    "AQACAwQFERIhMQZBBxMiUWEycRQVQoEII1KRsTNyFiRigqFTYzRDRZPB0eH/xAAcAQACAQUBAAAAAAAAAAAAAAAAAQIDBAUGBwj/"
    "xAAxEQACAQMDAgUDAgcBAQAAAAAAAQIDBBEFEiEGMQcTIkFRFGFxMrEVI1KBkaHBM0L/2gAMAwEAAhEDEQA/ANmQhC70eLQSZSpp"
    "5QNIXKMpEIHgXKMpEIDAIQjsgASgpBuhA8C53QThIhAsAhCEAJlGR3KbLIyJmt5AaO65n114r2vp6nkgpXtmqgNmtPB+VZ3l9StI"
    "OdWWEZXStHutUrKjbQy/2OiVNxpaVhdPKxgHOSue9T+MdgsZdHFKJ5R2YcrzjffEbqK91Uj5Kx7WOJw1pwFqskstQ4ume5zjvkla"
    "NfdZTfpt1j7nZtG8KaNPE76e77I7Vcf4grpJUu/B07WR5/UsFWeOPU1S1zWSNZkdguWaRndOawnZoWt1ddvZvmbN9odHaTQxtoo2"
    "G5db9Q3GUulr5sntqwqkPUl3h3bXTZ/3LGiJ4HqCc0NDd25+6x8rqs3ucnkzcNOtoR2xprH4M/B1x1DDs24zD/uVn/j3qMDULlPn"
    "/ctbFO1zQ7OxRoDTgqavq67TZRlpFpLl01/g3e2eKvU9BK1zq18oB3Dt11fpfx3t9Q1sV3jML+C8cLzdhpOEoj3xqIWQs9eu7Z5j"
    "LK+5hdU6N0zUIbZ00n8rg9r0PiD0zXwiSG5QkHf6lcj6vscsmhldCT/uXiJlVUw7RVD2/YqzFeK+F4eyoeCO+Vn4daVUlugjSa3h"
    "Lattwqs91U9ZBVM1QyNcPgqx2XkronxSutnukbK2Z8tOThwceF6esF+o75a46ulla9rmg7dltek65S1CPHEvg5n1R0fc6HPMvVB+"
    "5l0FGN+ULOGnglykRnfCBC5SIQgMAhCEDwCEIQAIQhAAgcoQOUCHIQhBEEIQgAQhCABNPKcmnlBJAhCEACEIQMEIQgQISgJO5QAI"
    "QhABtwU1zmtaXE4aO5QXANJK5h4neI9N05bJKSmka6qeCGtB4VnfXlO0pOrUfCMto+kV9UuY21ust/6Md4p+JcNopH2y3yg1LxjL"
    "T9K8y11bVXGsfPUTOeXHJJKfcLhV3S4y1dTK573uJyVC1jvsPdch1bVal/VcpP0+x6n6Z6at9FtVSprMvd/ciEQAUsUBdtjZP0tB"
    "wgvcNgcLEmy4AxxtOCNSMMb9DcFJrPdICTzgIyLAai055+6XzC7bATTv3wlwOyWQwBd6dHHylbnuQQkJGwym+kO75Rke0UkBx2Q1"
    "zc77I2/co0koyG0HY78KNxHY7IkJAwVC53ZGRbSXzcDZy6X4aeJcnTVa2mrJHPpnYBaTwuWE+yYS4O1Aq7tLupa1FUpvDRjdS0yh"
    "qFCVvXjlM9v2fxDsF2DG09bGXn9Od1tcU7J4w5rgQd14ApbxV0czZIZnsLeCCu6+GfjJolitl7m1NdhrZXH/ACt+0rqqNaap3Cxn"
    "3OJ9S+GcrWlK4sXux7e56OGeOyVV6Srhq6Zk0EgexwyCFYW7Rkmso4/OEoPbJYaBCEKRABgo4QjsgYIQhAhMJUIQAIQhADhwhA4Q"
    "giCEIQAIQhAAeE1KeEiBoEIQgYIQhAAOU5IEqBMQ8JEFCBghB4WJv95p7JZZq2okDQxpO6p1akacXOT4RXt7edxUjSprMnwjA9fd"
    "aUvTFhkkdI0zEEMZnkryLf7zWX69S1tXI5xc7IyeAs71v1bVdUdQSzySO8hhIY3O2FqDnanErkmvazK/q7Y/oXY9SdFdJ09EtVKa"
    "zVl3fx9h7AwEFwSulB9IGPhQ6iXblNcfVla/g3jsS5GQgkA7FRcpwAwhjHEjuEZyOE3GNwUA5UWCJWtzylLABnIwodRDgcp2oOB7"
    "JDwBxnYpOTlJozsCCUrAQgMCEjlPbtwlAbjBCNIDcboAZIRnBCa6Jr2+nlPz6ceydE5olGf7IAqPjcwkEBQu23KyD3nUQQMZ7qCR"
    "rHN+lBHCKZALc4SRySQyCRhIIORhSFuDhNIUoywQlFPhnYugfGqtsUEdBcmOqIBsHZ9TV33pfxDsnU0Lfw1S1shG7HHBXh9oLTkF"
    "ZyzXuttNYyqpJnRvac5Bwtn0zqa4tWoT5ic96i8PrHU1KrSWyo/ddj3o0h4y0gpR7LjHht4uxXiSK13RwjqNg1xOzl2WNwkYHtOQ"
    "RnK6VYahSvafmUmefNb0K60e4dC5jj4fyPQhCvzCAhCED4BCEIECEIQMcOEIHCEEAQhCABCEIAQ8JEp4SIGgQjgboQSBCEIBsUcI"
    "KOyRBEEcoScZQxjZXiOMuccALzp40ddCoqHWOjkyB9ZBXRPFDrUdPWSSKnePxEgIG/C8pXKtmr7jJUzvL3vdkkrQOq9aSX0lJ8+5"
    "2/wy6Sba1S5XH/yv+kWvDHA8lV84GVI47KI77LQDuSYuyMIHCUBAN5EAT8e5Tmty5SBhPbKWQRDjdDQ4OyrYpsD6hhNdTvaMgghI"
    "qRK7z3ICaC33AKfLluxbhQEByBko0kfVunNc4bA5CiEfynBjgNigME7tAdgsx7FK5mMYwQVGHEnDht7p4dkY9lECJ4wMEYUfyFNK"
    "M7qEd0AP1Z7JHNa5uTsR2SDGEZ5QBXe3Lkwt3VhzdtlC8HjCaE0NDcj7KRg7JIxlSFhG4TIYwWKGvqKCujqIXFr2OyML1L4VeI8f"
    "UNEy3VcmKqNoG5+peU3AEcbrY+hL3JZesaOpEha3zAHfZZrRNTqWVwsP0vuap1d07R1iympL1xWUz3EDkZSqnbKltXbIZ2HIe0EF"
    "XAuxwmppSR5Nq03Tm4PugQhCmUgQhCADdCEIGxw4QgcIQQYIQhAAhCEAIeEiU8JEDQIQhAwQhCB4BIOUqEACxl7ubLZa5al5wGNJ"
    "ysn2XMvF+5/hOlHxtfhz9uVYapdfS206vwjN9Oad/EdQpWz7N8nA+v8Aqia/3+R5eSxriGhaXI1ukPPCmmk86qe4+6ryOAJaRsuJ"
    "Vasqs3Ul3Z6+t7aFtSjRprCisEJJdsNkmnhAB1JyiVxW/UnhuAkY0YyVIMNOVFsMDmgacIdOIwds4SB/90fhHyN8wghvuouSRNRb"
    "7CCr1HBynF7g3fYfdXbd0/cLhMG09JLICeWBbHH4Y9TubltG/B3OrdU3Xiu7K8beo/Y0wu1DBITTHnhbfP4c9QwP0/gXvJ4091jn"
    "9H9Rxuc02mrGnnEZKXnx+SXkzXsa9pI5T27EgrLf8MXx3/02r/8AtFWqboPqyqcDDZa17T/7ZCfnR+ReTNvsYHUQMtOQjLSQcYW1"
    "yeGfVsMYfJZqto/25VGbpO7QEiqttTGR3MZCXnR+R/TyMG76QRuoyBlZCa01kAJ8mTA/6So2UbntaWEEntjcKammQdKS7opEJQ06"
    "c4WT/KKl7gfLIynG01LHEPjOAjegdNmILPgodHmPOFsVuszqmoDSCW53UldZnUznnQSwd8KPmLI1SbWTVmswQWhWWta4YTpo2wyZ"
    "zsfdR69JzyqieSi44IJWujfgn7JInmKZsjeWnKsSuE0B23HBVEuLXAFSXDIP4PZvhHfTeugqV73ZfG3Qf2W/53XCv4dq0zWGrpy7"
    "/TfsF3ULtGhVnWsqc33weSOtLNWmsV6ce2c/5FQhCzBqgIQhABjKEIQAoSpBylQJghCECBCEIAQ8JEp4SIGgQk3SoJAhCECBB4Qh"
    "AxHH0/svPHjjdT+YR0QeQ0Dhegql4jp3vPAGV5F8WruLl1nM2N2prDpWodYXCp2ip/1M6n4U2Lram67XEEaJxId9ioHg6ypG7jjh"
    "K5uo5XLkejsZIQE4A8hSMjOrdWBSvI9LCR8JSkkSjFvsVCMYSHIGVbdQztPrBaTwCE2O3VU0rYhE4Oce4KpOqiqqMn7BbqZ9ZVho"
    "btnhdn6Y8P4rkyJkkJedu3CpdAdAyPlZNJCSXkBoeP8AyvUXSPSMFsp4wW6nEZJIVlUqubwi9jGNCOZdzC9K+G1JRUkbW0sYGMnL"
    "VuTOk4o2eW1jBnnZbXT0rYoWhoU3lZG4Uo0Vjksp3c5PuaNL0bSveHGNoI4ICfB0hTiQgwRn7tG63cQt07hPbEMZxwn5MSDuJmpN"
    "6OodP/wkQ/7VYi6agiH8uGMD7LaC0dgm6fsn5USPnS+TV5rBTPBY6Fjs/Cx03SVsqMsmo4j92rePJaN8BMdCxw4CXkoFXkvc5Rdf"
    "CaxXIYNHG3PJaFotx/hxpJRLLbp/Jed2h3GV6QZA1u4UmGAY0hColRXtSJ4Q6r6D6p6RqC2vtMstMNhUQ+puP/wtUjkM0mQQR9Jz"
    "yF9Da2go6qIxzwMkYRgtcMrnl78J+ja2Z8xsdNqdkktaGnP7KL3RK6uYzXqR5Xs1JGwiIROMrgBgD/yslcbEJqPLGlzv6V1u59IW"
    "eyVQZQOjhd3hn2Lh8Z5Wr3Sempy6KniiM7m6GQxnUfuccBU97bKm5YwjgF+oBEXtAwWnhayXb6eCF0bquiMVQ/WTr/Ufdc9qIiJC"
    "QO/Kv6Tyi3qr3Iy4tAx3UL+QnuBzzwo3ncKsUGd3/hyuDWXetoid3AOXpccLyR4ATeX4iOaO8ZXrYcj7LrHSVTdYpP2Z5j8UKCp6"
    "w5L3SFQhC2k5uCEIQAYyhCEACcmpyBMEIQgQIQhACHhIlPCRA0CEIQMByg8oHKEACMZR3Qc9kDMbfJDFZah4PDCV4n6kmdP1NVPc"
    "eZD/AJXtXqAZsNSPeMrxLfxo6iqQf6z/AJXOut291NHePB+K8qu/fgpkDQEN2dumhwLdKdGMyAHdaH7Ha18GXsdoqLzcWUtOwuLn"
    "AZ9l6M6H8FKFlOye5v8ANeRnT2C0LwkoqZ95ia8NB+okr1TaYg1rA0YaAsdXqNvCL+mvLjlGpN8GOmHZdJRseSNjjhRR+CNghqRU"
    "xxfzRsHc7Lqse42VyFuQQQqKp5Kcrma9zULH0LSW2ZspY1zgNtuFucFLHG0AN44UjG4GFK0EKvCCiWlSpKb5HtGBhL2Tc4SgnhVc"
    "lLsJyU9uMYTchAKBjjuE0HHIRndBIRkWUGUZ2TeTtwlzkIyRyNdkDZRk98pzicbKJzuyTZJDXkqJ/GCpSQVC88qMuxJGBvVhtN5i"
    "MVxoop2jjW3OFrx6Qsdtp3/grdBHkHOG7rcZ3YKxlYQ5jt8ZCt5IrxbPKvirZfwtze9keGncYXEaxnlTPYRg5Xp/xetzxSOqNOdJ"
    "yvMl1fmqdtgq6tpZWCtUXpMblhjc0s9ednZ4CgkG2VLjLsqObIAHZXZaSOl+A8rWeJrQf1RkBewGnLR9l4w8GJfJ8UqMf1ZC9nM3"
    "a0/C6l0bLNo19zzh4sU8anCXzEchCFuBysEIQgAQhCABOTU5AmCEIQIEIQgBDwkSnhIgaBCEIGCEIQAIQgcoAxl7bqstQP8AoK8T"
    "9VN0dTVTf/cP+V7dujddrmA/pK8S9ZNczq6sB7SH/K571sv/ACf5O6eD8vTcR/Bhx2U0R/mBQb+XseVkbVQzVtzgo44yZJHAAEbr"
    "n83hHcaazJI734K2KWojdcntJaCAF6VtzMQNAC0zw36SFh6TpaaRvrLAX/crf4GCPYLEvl5LytJYwi7CNgshENsKjHjAKuxk5VaK"
    "LOTLTQMJdJTW7NUwVVFIjLSkUmN0mjBQ0DG42RuOycG+42TiExNkSE8jfhMPKCAuB2TcJTwgcZQA07KB32U5O6heNkmSRESFA/hT"
    "OGMlQSA6SoPsVI9ypPwsbUDOyyMmd9ljpRlxVvIqxOc+ItofcOnqhoAcS04XjK/wSUl1lglBDmuI3Xv+4UjZ4XscA4EYXlDxr6Jl"
    "tl0Nzp4XeTIckgcFVLee2WGXf6oYOMA+oIeRp37qPJBwSnk6o8HsskWDfJuHhO4s8Ubbju9e2Iv9Jv2C8WeEMHn+KNBgZ0nUvace"
    "fLb9l03ovP00/wAnnnxda+vpY/pHoQhbqciBCEd0ACEIQAJyanIEwQhCBAhCEAIeEiU8JEDQIQk3QSSFQgcIQDE/UgpUIDJBUM8y"
    "ncz3GF448VbY639e1bNOA52ofuvZpAwvN3j7ZvLvMFwY3626XbLUOsLbfaKou8WdT8KL90dTlbvtNfscQaCYhjt8rrHgV00epPEa"
    "J8zXPZB/MJO/HC5W1oEWOCvV/wDCzZoxaqu5OjGsuDMjlcnqvjB6VpcZkegI6I0tMxgHpA7JGx4dkLKzAFmnGyqtg9eDwrPbghvy"
    "EDDgLIRMIWJuF3tlmh11lTGw42bncrV2eI9PV1joqZrWxg/W53Kako9w2Slyjog5UgGcY5WpwdVUmgSPmaMc5KydD1DQ1frZUMI+"
    "CqkZRZTlCSM5pKUNIVNl0p3/AKxj7qUV0LhgOCllFJ5XcsObvskGc4TRUMIGSE8SNPtlSwRyIQo3NHPdSFwJ5Ub3NB5UXwLJGkxu"
    "ldJHxndMMsYbklLJPA/SccKNzDq4Vea7UtM3+bKwfcrFT9XWtmcVMbiOwKTlEkoyfYzJiJ3AULosOw4jBWm1XiDTs1+X6iDsAVjH"
    "+IdLPE5s75IP+v2VJ1IlWNGffBv01I4jLdx7rEVMTmOIK0mm8Ra63vMhkbW0QOSeCAs3B1/07c9DfxLYnycB3v8AdUnJPsVdk4+x"
    "fc0k5K13q3pei6ksUtJVRtcHDbPZbU1rHx6mODmncEKORg0n2SwSjPD4Pn/4idH1HSXVEtK5v8onLD8LVGOy0gr0z/EPZYXQRVuk"
    "B2cZwvMcuY5S1ZGhNyiQrrDz8nUvAildP4mNkDdo4juvXrfpH2XmD+HmCNt7qKp2MkBoyvTzXBwBBzsut9HxSss/c8z+K0pPV0n2"
    "UUOQhC205eCBshCABCEIAE5NTkCYIQhAgQhCAEPCRKeEiBoEIQglgEIQgQIR3QgeBHHDcrjvjbRGr6fEoaTo7hdhfnScLAXOy03U"
    "cc1rlwXOYfSfstf6lw7GcTevDtuOs0pJZPEbpA5x989l7X/hkpyzw5L9ONchXjzq2wVHTXWVXapxp0SENyORnZe2P4bqR0fhVTSn"
    "PrcTuuL1fY9Wpels67JCclaJ1v1XVWOiNNboS+rfs12Mgf8A9XQK1xjpXuGxAWnUtoNbdHVFcwSYd6dQ4VpVyuEU6WM5l2OPy9M9"
    "e9XzNlqC6niO5c47lXWeGHVNooHSU7GXOYbsaDo/Zd6aKWkgG7WADjhaN1T4w9PdOTGhieKutO3kxEen/ceArfYl3ZewrTk8Qjwc"
    "evFR1faoddz6WuTMDB8oamj52Wvs66qaeMsimMDs7tkBa4fcFZ/qP+Iz8ZUS0cNDmpiJDo4B5wAHckbLmN28W7NeqrRV00MpI3kZ"
    "FghXVOjLGcE/qILifc6DbvE680Ujf+aE7Pl2cLpPT3iTBcKceZVMbKNyw8rzZBU0NwYZbRV00pI/0caXLJ2mR8dbHs+mmJx8EqMl"
    "tKyoU6yyj1nTdSOnLXxSNkjI2+Fk6TqFr3mOUOa4LiFmrbrRwNd6nDkgLdLdcpayIEBzX8OBG4VHz3ktKloonSBd2ufkPyES3aJr"
    "fU8DK1OGmq5CJPOcCOyrXP8AGRQF2HOxxjupOq8ZLdUYt4NlmujWvz5u3stY6h64ZQnymv3A23WryXavdkNik1cbrU77SPn1z1bp"
    "NRHZUfOb4L2nZrPqMB1V4iXee5SNp6p+ckaFrh6muZjDH1Mhc7cBnLisLfq9lDLLIwsiiacGWQZJPwO5WswdV9QfnDbfYLTPU3B3"
    "qa18fmPAxnIaONld0aEqnKJ3FSnQ4Z1u02jxCv8AOxtAxlLCf1zOx/4XXLD4S3uamH5zeoMOHqbCzc/uV5ZoPEzxPisk93bHVGhg"
    "mEEla2EiOJ5/S48ZXT5fEvxY6Pt1Hceo7TVtttQGSMulPmSB7Tv+2ylO0mucFmrtT4i8HfH+FFoZT6IZ54zjB32d9wtNuXhMaOtf"
    "LRVczmuyfL4AKl6c8e6SqjgdcWtkpZPpq4d2j4cOxXUqa80F2pWVFJNHIx4yC05VliEngnJ1qXMuxqXS8dbb6VlvrteW7BzjnIWz"
    "SQZbkcJ1TbtRE8O7sq0yJ4pwHYzjdV4xaWGW7nl5PPH8Q0LW9MscR+peRp2OnqgyJhc8nAAHK9qfxBW78R0DLK0bxnK87eC3RlR1"
    "X4q0rDAX0sB8yYkZAAV1Qkopsqyj5mEdW8Eehaq09L/mVyhLJZvU0HbAXVqKtjlq3QxPDg3nB4WZvtvMFsdQUQ8qNjMEtHb2C07p"
    "KmbAJ2jkPO/utp6Nv6877yVL0/BzrxU0qzjozu3DM8pJ+5tgOQlSDhKuxo8sgk3JSoQMEIR3QHcE5NTkEWCEIQIEIQgBDwkTjwmo"
    "GgQhCCQIQhAYDukKXukOcIGDsaeVRsdHNUdYmriDtMQ0/wB1bme2OFzieAsx0bGI7e6qe3Be4nJWkdcXapWSpp8yZ1zwf0519Ulc"
    "NemC/wBs81/xR9NR2zqu2XyOINFW3Q8js4H/APS9J+BlIKXwbs5xgyRaz+60fxxslL170Yae26ZqyhmEugcgcFdc6Dt/5X4f2mh0"
    "aDFTMaR86VyiEtySPSFzDZDc/cz9QzWMchY+WIQxlzBusuwB2xCSSma9pbhEoNmOUsM5L18691NqkhpKp1NDg63xj1kew9lwi39A"
    "Xjqq7w2urbJbLRNLh88n+tU4+TwF7CkstPI7+bGHD5WGuvSdPVPZNktdE7XHp2wVaqlKEtxk6F3T2+W+Puco8X/DnpvoL+Fy40/S"
    "togpZ3+WyoqmNzI5pIDiXcryD1DVUUVis1spOkaOiNN6qmuglc+atz/UTs0Y7BfQy6VNRUdPVFhv1pZc7dURmKRvB0kLzRdv4dmS"
    "3V8di6rmpbcXamU9XT63xD2DgdwsmrpYWODHq37qXP3Od+HPSVr8SfGR1ksVnqbRbZYHy+UZzI6mIaMODvv/AJW/z+HvUHSvVDbV"
    "eC+uodYbFXgbt32Dvj5XW/C/o+g8LLZUR2iidcLlU/69xqTguH9LQOGrZKxnUlzqXPNFbXRv/wDUa52lUbirCcMLuXVs50p5f6TA"
    "0fSU9NRRyDLhgbLaLHaoyQTEA5ZyjppobaxlRpLsb6RgfsrNDT6KjIGBlWMaXIqly5Jk35IPKDm7DuFUq7Q0ReoZW0Q/6eCoKuNr"
    "4iHBXcqSwWUKryc7l6fifUHQ0DJ3WMvnSUc9A4NZlxGMdyt/MOl+NKiuFuFfSGM6m/LTgq2VFF3G6lGS5OJ9K+D9puHUJuPUcHny"
    "xPIjpXjDI/27lcH8Wemr14f+NFwkpaqqtUddqkpa2Bxj1McMFmof2wvYlfYHzSCU1dW2YDHmNkIJWvXvombqGh/L72G3ekzlsdWw"
    "PLfseQrmFfZHbgnVfmz3t/2PFdhkvzbPWdIW27VdZFd5Wj8shdqZNJnZxHuPdfRKKzWqw/w42fpLqCOCd0FAyKaJ41anadwFoPR/"
    "hN090tWfj7HYKagrP/X06nj7E8fst4dY3T1AlrZ5J3f9ZJVZXLUHjlss501KcfZI852/weq4LrU13TGqihkkLzRTjVE9ufbsus9J"
    "9PVlmgaySBsBP1MY7Lf2XSoKKKNgY1gAHsE59Mwn6Qsf9PmW5vkva1+5rYlwUoGExjP9lI9h0nAVjydO4SEZafdXKiWO45Z4tWo3"
    "Dw/uULW5cYiW7d8LT/4cemYenfD+e9VUYFbWvJJd+loOAuy3ukiqaCSOQAtIIIK0Oenqp+nJrNZG+SYm4OnbA+Fb1JuCwjIUEqmE"
    "2brV/hquhfHHKx7nDkFc3tH/AC18raRww5r8gKKxzV9re2OeR5IO5cVeudOI+pqe5xHDJ24cAsv0rf8A0+o05Nd+P8mD6/0R3ug1"
    "6UX+lbl/YzI5SpGn05Srv6PGDQI7oQmIEIQgQJwTUo4QJioQhAgQhCAA8JqcmoGgQjuhBIEJMd0qBAhCMZcEDRWqG+bLHCeHu0lZ"
    "u6zG22JlFTDD3DssM/H46n/3rYrnC11XTvlHpwOeFyHr+pKV3CnnjB6d8F6NOGm1K2OXL9jXenOlrk+6/mpJaz9YP6h7LrVHpbTM"
    "Y1uGgYA9lXoaeL8oxFgZHZWYGlkTVqFGmoI6Xe3DrTLbfqU4OAoI8nlTtCqmPYFu6ikiJGVa7IxkbpNBkxr6NsrS17dlF+TUZGDT"
    "Md8kLLhoztumnYKOxE1NrszDOtEJ2EbWj2ASCijiGGhZZygeAQjakPe33MVJECOFNTRBozhSOaNXCkY3BUUiO72LcbcMwo525jJU"
    "0QOhMmaQCFOXYglyYxzQSMhTMYC1NOclSRnfChEbA0jHDJG6RtEwO43Vth2wUuylhC3Mh/DDGE00wyrRKacEpPAJsrmMtGyDGC3O"
    "FNsSkI2wgeSq9no2UBGGlWnDYgqpO7DUmSRibgwPjc07KjZrPHSyTTtcG+ad8q5Vu3JKrXCtjoum3zOfodjbJVGpjuy6pbn6YmBv"
    "VrpRWOILSSeywl7j8qGjb/S4YVeyVF16h6gcRkUzDu4rIdVGJtdTU0RB0uGSFLS/Vd0mv6l+5d63mlptxGbz6JfsSM+gJya3ZgTl"
    "6Pj2PCM/1MEIQpEQQhCABKOEiUcIExUIQgQIQhAAmnlOTTygkg7oQhAAhCEDDujujukPGUDIZvTVQPPAeMroMtqju9gLCD5jWZY4"
    "diufVQLqYkcjcLovTlwD7NCS4aSMFcs69tv51Kt7NYPRPg1fZsq9unzFp/2Zg+lb+6nqpbJXnE7CQwu7hbhBJqjwcZWBvHQ4uNX+"
    "Y0VUIpm+sHKyNvMkbRFM7U8DDj7laJSzH0yOvXXlz9UDNRHIVkKrCeynBwqxj2Tt7Jf2UYcSMqUcJiGnbgYTCeyeeFG4d1FgRuyV"
    "A/0hWDlQSjdJjyVwCZN1M1mZMIY0+ysRM9W6aQmyaOMgcpJo9twrTIgGDdO8oOYfhNogpGBlic15wNk1gw7KyUsXOyovYWPwQqb4"
    "JtkzeMpeySPGE8jGykREyjTlGCncKIDNJHdMc7HZPcSqzz7lBJDZHcrH1DzhWZHYaqE78jCiTRja1+c4WqdUUl5vEkNut8LvIaPW"
    "/tlbPUaRkuOMFQC5imJ0Yz7qjVjuWGX9rJwluSE6esJsPT7hKGh5G61O7xiTqCOPnHqJW01VdLLSGWeXEbd8LVYXOq62WscMgnDc"
    "+y2DpLTndahBpemHLNS8Rtb/AIdolbc/XV9K/wClv2CVGEBdzR5DYIGyEJgCEIQIEo4SJRwgGKhCEEQQhCABNPKcmnlA0CEIQSBC"
    "EIBghCMZQIa4AtIKv2G5GhmNFM7+U45aSqWNlFM0lupuzm7hYXXNKhqNtKlLv7fk2zo/qOroWoQrxfpfEl9jo9LUziT0PJaRsnRN"
    "kErnuaRusP09coqmiYS7D2bHJ4WenmYXBwkBz8rh9SEqU3Tn3R64hOFenGtSeYyWS5DJkAg7q6DlYyJxG+Fdjk1YKmi2a5LbfZPy"
    "VEzcqZMixMklIQpMNwmHA3UWQIu/ChfuVYI7KGVuBlJk0PjaxseXED7qEXCmZUCMzMB7DO603r69X+h6Zn/4co/xNdpIjbnbK8d1"
    "vRPjveesjXVE90bVSP1CUTuaxgz2AOAFFPc8FzG1bhvPoGanTvnZNNeGsPqwubeHw6ttXQdLburq1tbcIxgzjlw7A/KyfUBu9X07"
    "U01oqPw9ZK0tjlxnQffCUpNFFUluwbQ28UklYIJKiJrzw0vAP9ldnjjfHqa4HbbC8IdQeCXi5H1VJcYrlcK2Z7y/8UJ3B3+dl6L8"
    "IajxAtXTot3WrjM5m0cjzlxHyot7S6nZYg55X+TrMez8KUjKhpsyNEh2yrWkYU0Y9jQ3bKCNlIBgJj0MCJ42VOb2VuR2yoTHcpMk"
    "itKeVQmOSR3V2Q+lVJS0cqJWgjB3UyCBxbyBssPQF5jMlVgHnCf1LeDHXMpId344Cwnk10x/mTENPbKyNhoF7qXrox9PyzHaz1hp"
    "WgYp3k/W1nC5ZPdauSvqBRwOIjB9ZHCmijbFG1jeAEyCBsAw0ZJ7qbcbELrvT+iQ0qh5a5k+7PM/WvV1bqK88x8U4/pQqEIWwGlo"
    "Qo53S/ZCAYIQhAgSjhIlHCAYqEIQRBCEIAE08pyQoGhDuhCEDBJylPCQZQNCoQhAcAkxylQgCEMfGT5Mj2A84KmbV3GIZbVOON8O"
    "3QkO6xtfSbOs3KdNNsz9p1PqlrCNOlXkorssm+2C4G4WiOZx9X0u+6zsPC0To6rMdbUULjs71tC3eP68LjWrWf0d5Oj7J8fg9TdP"
    "6ktT0yjde7XP5XcvNO22ymadlWZypckLHsybJc9soUeRqyEutIjgemSN1NKXWEjpG4IygaMBXUFU+p1QODT7kZSQtkpQfPLXu98L"
    "I1FU1uR391j26ppM74SSwVN7a5LAkglhy5u/ZNaYWnhSfhy2MkjCgbC5z+dkSXuJJDJpnuOgO0+xS01seZfNlmLhyAUTNDXYI3T4"
    "ah0Z33CisPuNvjgy8DQGgDsp8KhHXxFozsVMKgHuEyk0WCVE84ykEuW8qNzvScJMaI5XYbklUJXnOcqxM/KoynCgTSIZX+nCpVLt"
    "MRcTsArTvUSsXe5xTWiWQ84wFCfPCLy3S3ZfZGljFVeKiqfuc6Wq1jblVqBpFNrd9TtyrS73o1qrayp0sey/yeO+q9ReoarXuG8p"
    "yePwuwh2QBndKlwsqka7kQDdKeUqaUxLkEIQgAQhCABKOEiUcIBioQhBEEIQgASO5SpCgaEQhA2QSQIRjfKEDBCEIEHdCXAQQgWR"
    "DwkwUqEDyPpZzR3aCqHZ2HfZdTpiJqdsse+QuTSN1RkfC3zo65mptzYHP/mR+ly5z1vp/MLuK+zO7eEutKVOpptR894/9NnbnbbC"
    "kzso3ZGMFKCC3BK5+jsbQ9mSVHI8MG5490jnhm2Vj66pGnnCi+BJEktaQctcNlCKpzmEjlYSWvpoyXSTsaB7lMl6ktNJD5s1VE1u"
    "O7go7ioqcn2RnGwyzguwrEbKel3qZmt+MrRa3xLtoYIaSsgGdtWsKnB1HQ1Muqa4xyye2sKccMuo2FZrLR1IVFDVQmOKdoI90wOp"
    "KGPXNM1xPABXPB1BTGYxxSsaQM/UE2bqOnaAZJm44wXBVWlgirCrk3Wa42yofjV5RPBPCaabUzVFIJB7grns3UVE0kPqGHvgdlgH"
    "+JtJbKxzYq14DeW6Tv8A+FbScUVf4fUf6UdXn8xjctB25CIKx+n2XLHePfTEULjcpvJIG73MIH+FiYPH/o2uubaSlrXukcdvQQCo"
    "OWOShK2qLho7rR1bjtIc/KvGQFhwVq1iu8Vyt7KmB2WvGQs7E4452QnlFvKLTwxZSqj99lYlcPdV8ZOSgnEiLQN8rTer6syGOiYf"
    "rODhbjVOEdO54OMBc6mkNZeZJ3ElrDgLL6Bp7v7+FP2jyzA9XazHR9HrXGfVJbY/lkkbQ2JrR2CejlAC7skksI8gybk8sAE5ACEy"
    "DEKRKUiBoEIQgYIQhAgSjhIlHCAYqEIQRBCEIAEhSpCgEIhCMYQVAQhJlAsC90oSDhKECYqEIQRGnlCUpEEg7K3Z682u8sm/+U84"
    "eqm6aW6hgqzv7SF3RlRqdmZbRdVq6Vdwu6PeL/0dhpZmVdO1zSDkZSOGl2FpHSd8fFUChqXk/wBBPst2mOcPbuCuJ6jYVLGu6NT+"
    "34PWGi6vR1e0jd0Xw+/2ZC92Rutdvs00dK/yWlzsbD5WelB7KjPAZTx/dY6SMrF4eTzl1J4cdZdZX/zam81VHSDOIad5b/fCyVg8"
    "IIaN7GX6ora9zHbedO4gj7ZXd46RsW+kZTayCGeMNkbgjhwRHthl7C8cX2MBbvDXoSSFrRZImP7d1novDbplrGgUMQA9mptO6SBo"
    "aJCdPBWQiuR1H1uB+VWW34K6vq+MKRja3wq6Vry0yUZGD9THlv8AhWKbwz6ZpAHR0jX441HP+VkBc3jJ80/ZM/My9uXSOafgp+ki"
    "7y4aw5Dh0f0/E8PNLADj+kKnV23p2np3tbbqeSTsSwJtVdX6S1hPxlYt73zvzLL+2VByj8FJ3FV/qkYG7dI2rqMugq7ZS+WecRhY"
    "WXwh6Wi8t8NDE2RnDg0ZXRYZGRxBsTR9044dz3VCeHwQ+pqfJr9itDrPEII3HyhwCtnifhgHfCrGNpGylacBRUcdijJ7nlkziXco"
    "a04ykG4GVUulxhoaBzi8B2NlOMHJ7YrLYnKMIuU3hLuzD9T3IU9I6niOZH7ALVoIhDFpOMnclPmmlrq41U3/AGhOwuu9KaI9PoeZ"
    "VXrl3PNfiL1ZHWbtW9u/5NPt938ipQkSjhbcc1YqEIQREKRKUiCSBCEIAM5QhCBglHCRKOEEWKhCECBCEIAEhSpCgEJ3QhCCYIwE"
    "Y3yhAAlCBylQJsEISEoEhChCEEsAhNc4N3JVuC21E8P4iUinh/rfsrW5u6NtHdVlhGR07SrrUJ+XbQcmVYnvZcIHQgl4kGMLrVM9"
    "slKzOx053WhWOKideRT05bM9oyXnst1cHRsGDuFyfqjU6V9cJ0lwl3PSHQPT9xo1jKncv1SecfBNI0nYBVnAjdSRTtky1xw75Q5m"
    "RzgrW/Y3dldx7KCRxIx2UrzkkNUXllx3UWiSMbUmRpJY8g9li6jqKooWkS0JlA7s5K2V9G2TAKY610+DqYHZ+FB59itGaXc0yXr+"
    "kaT5ttq2f9uVX/45qJjigsdVLnhzhpC3M2K3vlDnQMOPcKwLZSMZhkLG49goPf8AJW82n8GjCu6jrXh0lM2mjPYblZighfE3+ZqL"
    "uclbE2jZngYSmnjBIDRgpYl8lKVRP2KcLjgBTN3cninHDE9sPlnJOVLDKTY5rdt0FvqTgchNklZE0l/9kDQr5GxwF7zjC57cK59y"
    "ushJPlMOA33W6Th9UwkjDMbNWkVNuq6GrlmYNcTnZIby1bL0pOhG+Uq/xxn5NL8QqN7V0iUbLOc8474AY2GEqjjmbIMgp+/suzRk"
    "pLKPLM4Sg9slhipQkQpFNochA4QgiIUicmoJIEIQgAQhCABKOEiUcIExUIQgQIQhAAkKVIRugaEQhGEEgSjCVCCOQQhCBAmpx4TO"
    "Agkhe26a5wB34Ts7ZKy9is7q5xr6gYpY98H9RVjf31OzourUfYzWhaLcaxdRtqC5fd/CIKSgjhpXXS4NIjbvGw9/la/d79LcXFr3"
    "PbG0+mNuwwth6iuLp4Xwtb/Lbs0ewWiSyZkONvhcg1PUat9VdSfb2R6v6e0C20a1jb0Y8ru/ds2/oIyfnT3ZyC3vyumuJ4K590NT"
    "Bj3S9yFvxdlYaf3MxPmWUVp4ifUw4PumCuLP5c/PZwVzJOyo1MAewghQyQaGyy7BzSCD8p7appAHf2WFc6Wnl0kks9inmsa1mdx8"
    "p5FsM42ZIakH0gnKxDKqQsyDlS/iA1urPqSbHgvum0nJQajW3ZYuSrGMOduoBXNjf9WyjgaMwJnNcmGZxPKxsdwbJkNOVI6fLA/V"
    "sOyBYMvE8cHY+6ZPK0OGCAsU64HTpaTlEZlqHZcUN44Eo/JdfWDOiJupyWOEvOuU5PsnQU7GtwBurGA39lDJNEL2gZC0WvuzouoJ"
    "Ih9A2+63ioyGk5XLbtLp6gnz7pxfOUXFGKb5MxLDTVv8ynIin9hwVUbK9kpgnBbIOx7qGB5IEkbsEK8ac3ekc+MYnhGQR3W4aF1P"
    "VtZqlVeY/sc/6z8PrTVqUri1jsqrnj3ECM74UFLIZIfUCHN2I+VPwV1qFRTipLszy7Woyo1JU5rDTwxQlSBKplBgmnlOSEd0AmJn"
    "CEYyhBMEIQECwCUcJEo4QRYqEIQIEIQgAQhCADCEIQAIQhAAhGU0lA0KU0pUbAElJkkPp6WWur4qOEZdI7St5uui12eO2QADS3BI"
    "91X8ObOay8SXGRmY4hhp+Vlus7e5odOxuff4XNeq711q6oRfEf3PRXhho6s7L6uovVU/Y5ZfJXQ0z3ly019dFHIHPeCSdwsp11Xy"
    "UNt05xk41LWeibHXX+7Nqasu/DNdkNx9S1F4iss61F5WTtPRETvyps7mFuvcZW4LF22BtPSsjY3DWjGAsmDlWE5ZZTZJj05B3TNi"
    "d+EJjyQMhU8iK09PFLqJbn5CxE1P5biC3Le2VnMnCrVDWSHGENjwYdxDYcZ04PCYZiyDLfWDtlWZ6chpwMhYx0TskgEfARuJKOSJ"
    "x8x2lz8H2UohjdH63uz2ATBSPzrdsVLHFLjYJNktojIA0AtJBVmGEketxISRxvIAIWRpqcnBIwPlLJBojhpwdmj91kIoWswpGtY0"
    "bDCcGgnOUmyI/AB2RjlCOyRLBWqP9M49lxzqWofTdTyh2wdwuyyfSQuWdf2l9XE+ppzplYCQR3U4cMr0XyY+317SQ0nlbl0g/Xd5"
    "GctI3XFrVeXMkMM+RI3bC7x4cW18tm/MJGEukOx+FWpx5yipXmoRMLd6X8u6rqYBkMk9bfZR9wtn69tropKa6MZs30SFaw3Dmhw4"
    "XZem7rz7KKb5XB5O8Q9N+k1ec4rEZ8oVOTUoK2I0NioQhBEadikxunFJygnkEIQgGCUcJEo4QJ9hUIQgiCEIQAIQhAAhGdk1A8C5"
    "QSkQgaQIQmuc0clJvHLJRi5PCF9SMOkeyKNpdI9waGjurlus90u84joqR7gf1EYC6R0r4bfgqyK43WcPlYctiHAPusHqetULWDSl"
    "mXwbp030beanXjKpBxp+7ZsvR9mbZum4YHNHmObqf9yprvQx11PJE5o9QIyVnBEMkh7eO+yqvhOjJAIXLatSVWbnLuz0hb0YW9ON"
    "KmsJLCPH3ifTVUvX1F0jURaKeFxmLwCDJk7b910HpW1w0NGyNjMNAXROvOgaPqZsFxZGG11IdTSOXD2Ws2+AwR6HMLXNOMFWNyzM"
    "29RShgysDAG4U2PZNhaNOU9WLJjNR90hISlpHZNOFFjQvZQuj9WcqQkpMAjKRLBC+MHcKm+jOrPKyBOdsJhI90s4GY/8I1rskkpf"
    "w+HYDeVcOM7hPHuAlkCOGlAwSArOlrUzdPAyPlBDAcjYIaDjcKQNIxkJcYKBYG/uk7pVGRgkoJDJyA042Wm30ZY8uGW4K22Y5jIW"
    "o9QSsbTSZPZJFSn3OJXezz1vVsFNbmkT1EoYA35K9d9NWltp6doqHGHRxNaT7nG65V4WdGG4dTO6qrGHyoctgae5913JrSMLJ28P"
    "Tllnf1t0lFGMvdqZcrRPSPGQ9u3wVxjypaSpkophh8Ti0grvztxwuadf9PviqG3qkZtxKAP/ACtu6b1FWtfy5/pkcz6/0F6pY+dT"
    "Xrp8/lGne6Bwmse17dQOyd+pdOTyecXFxeGOBSpqUFSItCnhN4Cd2TUCQZQjGUIJAlHCRKOECfYVCEIIghCEACQpUhKBoRCT9SMk"
    "uDWAuJ7BRbxyycYOTxFZDB90heAcLYLL0Zer04PEJp4P63jC6FZfDW0UQbJWaqmXn1cLBXvUNrbenOX9jd9G6D1HUUqko7I/LOXW"
    "6x3W7SAUdI8tP6yMALpPTvhnRU8bKi7/APMS86OAFvUFFT0sIjhjihYOA0YKnAyNiQVpuodR3F2nGPpida0LoSx0xqpNb5/cq01F"
    "T0MIjp4WMaNvSFabgtGCkA1AhObsMcYWvtt8tm7JKPCWAdhvKVxAYWNxjlKw6icn+6a8ahvx7gJDKsg0nGn+y0fqe1fhqoXCBh8t"
    "x9YA4W9uw4Y3/dVauljqqZ8EjQWuGCoTgpRwTp1HB5Rz2mIc3Y/ZThm2FDU0MtruTqd/0Hdp+FYb6m4Cxk44eDJxluWSMt3wVC5u"
    "+VYc0qJ8bsZAKpk0yHc5SHYJ2DnKa45OFElkY4lRu237lSPOWjZIBnbCi0GSMEKRrkhjOduE5sbsgnhLAMcN909uUob7BSNCBCZO"
    "N0hORhKSc8JM/BQhZGk4UL3AO4Uj87qsQeCjBLJBUyYaStSuME1yvENupmF75XAHHYe62aukayNyz3RnTohDrvVR/wA6T6M/pCrU"
    "ae5oU6qpx3Gx2S1w2qyQUULQAxuDtyVkSAG7BKBhKVlorCwYaUnJ5ZGoaumiqqV8EzA5rhggqwkPKknh5RHHszjnUfSNwtFW+poY"
    "zUUhOS1o3atfZOx5wctcOWnkLv0kbXsIcNitQv3QlvuRdUU3/L1H9TeCtt0vqedDFO45Xycz6k8PKF7KVeye2b9vZnNdu6XA7Ke5"
    "We62WUsrYXPjHErBkfuqzJGPALXZW92l5Suo76UsnFtT0i602o6VzBp/6JBwjAScJcq7MUCMBCECDAQhCABCEIAEm6M/CAHOdpY0"
    "uJ7BRclFZZUpwlN7YrLE/dIMudojBc7s0DJK2eydDXe8Ykkb+HhPd3JC6dZOi7PZmsLKdss2N5Hb7rXtQ6jt7bMYeqRv2g9AX2oN"
    "Tr+iH37nNbJ0Bd7s1s1QPw0J3y7krolm6Cs1sayR0XmzN/W/dbW1rIxo0gjHZITh5bjstIvdcurp4csL4R2HSOjtN01Jwhul8vkG"
    "RRxsDGANA40qQgBmGj90050jBGELD8vlm0R4WEIW+kADhKcCIDgowc5z+yXnnCaQxuQDjul1DV7ppZl+o4SPnjhYdTmj7owA8HLg"
    "mk4JG+FTN0pzLgEfsrTi2VocBhGAEIB4CaWgnKc3LWoIyPZJoXcxF7tTLhRHQMSt3aVqcLHscYpGkPbsQV0EA4wSsVcLNHPKJ4yG"
    "yd8d1bVaW7kuKNbbwzXRD3xkJHwgjBGB8LJinLH4e3GOUr4WYyP7KzlHHBexnkwroD2AwoDTk7ELNyRNxgKB0IxjuqTiVEzDSQOb"
    "9IUGlw7YWcMIx7qN1O07aQoNDyYppOMYS4cTwsiKNncFKaUE7BGB5KLWOPb7pwhcXbZwsg2HBxhPMQA2GEYAx/4ZxKjdE4HdZLT2"
    "UZhGeMowRbMf5bsZHCrzMIBwFmHRtDdICpGnfUVDaeNpJcdz7BTjHInNRWTHWq1OvF2bqafw8Zy4+5XRY42wxCJowGjAUFvt8VBS"
    "tjjYG9yfcq07OdlkaVNRRja1Vzf2ExvlNOcpRnUlwq5RQwppwlcPlJ24QMQ8JvbdO7prgCECwVqmmgqYjFLG17DyHBaNfvD+KQOq"
    "rSTFIN9HYrfjsUvIwVc213VtpbqUsFne6dbX1N07mCkvucGqG1FDVGmuEToX/I2KcHAt23B7rsN56ft95pnR1UAcTw7uFzG8dL3C"
    "xyl0YM1L2I5C3zSOpYV2qVfiXycV6p8PaloncWPqh8e6MaOUqjjka4bFP1LbVJNZRy2cJRk4tcioSApVIgCEIQBsti6Eut4IlqQa"
    "aE+43K6XYuirRZmNIgEs3d7xlbJHGyOPS1gaB7J7flckvdaubpvdLC+D1No/Sen6ZFeXDMvljY442DSxoA+BwpoiA8CRgcPhMG3d"
    "KHbrEPlmypY7CB7XEt40nb3wjDt3ZOCopP8AX1t0j3zndPDiW7lAx3ATHSBh3Tu3Khlj1+oHcdk8AKaqMDB/umOrYQ0nOfsoDTHS"
    "c8n2UDqRzcHO/smgJXXB2fQFRqC+Z+XnKsMpXatyn/hd9yUCyY4Q4P0q9T1EkZ0nJCmbTNHbP3ThAM9ggTZIJwcFp2PupGzNI2O6"
    "gxgEY291CToOppKBZL4OeEjhvuqQqyHerP7KZlUJNsKLQ8kVXTte3WAAQsc4Aj7LNZDmn3WLqYXRTZ/Q5WtanxlF1RqNcFNw7kKN"
    "zRjICsPbuMqFwyMBWbRdp5RAWEo0AcqYjAAwmkfCi4kskRAzwjT+yf34SpYHkZoGUvl5TuPlPbtuhJBkh8kYR5WysA5UsMBmdk/S"
    "FUjDPYpzqKK5KkdHJMTpGB7q7Q26Omc553ce6uhoY0NHCdkYV7CkkixqVm+wE5CaeEEpFWKO4EITSQ3cuQPIh5TSd0478FNI3QMa"
    "c4RjZL2SHhADCAEnITiB3KQgAbIAjycqOWKKoYY5WgtPOyc8gnTqIPwkZnjdCeANF6h6FZK59Xaz5cvJYOCtHmZNST/h6uJ0Ug23"
    "4K7m/BHKw146fobvTujmiAdjZw5C2TSuoatq1CpzE0PqToa11ROrRWyp/pnJ2ke+QlVu7dP3GxzHUx01N2eB9I+Vj45WvZlpyuh2"
    "d/Ru4KdJnCNX0K70us6VxHH39mSoSfujkq9MPg9L5z2wEJ2Aoy/Dg0gjPdcNPZY45RwcpRghB2HKAAnJ2GSkIxyEoO+Bk/KDqxgE"
    "gdwgBBul07HZI0YS+onZNMCNo5ymkZPCfkkoIGE8iY0NbyUFo322S5wdgl5CCIwgAYSYT9O6b3wgRGR8KJzMg7Kw7fsmkZGEAUXx"
    "8gBI1rmnIBVzSPZJoB4SZIGHO/dErBLGWkfZDRg5TyTjZQazwSXBiZInRu0vG/ZREYCy8sYlZg7H3WLkYWP0P2VlVptF3Sq5WCuR"
    "vuE12OMFTO2OFGfsqOC4yRYKUYwnH7I0/CQZQgbvlPxvnCANk9jHOIaOSpKOSMpJcixROlfgcd1kmMEbA1oTYIRFHjv3UqvKdPHJ"
    "ZVJ7mM/ZIBlOLwFCKmPJAOPlVsFEeQRykUZqWHvlIZg7gJ4Fgk1DOMrG3GRw9LSffZXC4+6pzQukcTyflGAwUY6yaP8AUSPlXIbk"
    "07PaR8qu+mdn6VCYJATsjBLJmmTRyfS4FL9isE7zYzlpIT47nNE4CQZCMDMw4pueVHFURVDA9h39k4kgcIwLIxzA5wcMgpW7DdPH"
    "CCkQyMcARsN0zfOCnk+yaeed0DyQz08U8RjkYHA9iFod+6HOt1ZajpfyYuxW/uJB90haCNwrq1vKtrPfTeCx1DTbbUKTo3MMpnEC"
    "ZYZjBVRmKVuxDk8ELqN56doLrCRLGBJjZw5XOrpYrjZZCXMdLB2eBwuhaT1HSuEoVuJHDupegLixbrWa3U/j3R//2Q=="
)


st.set_page_config(
    page_title=f"EduPredict | by {DEVELOPER}",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


def load_photo_b64():
    """Use assets/profile.jpg if present (so you can swap it), else the embedded photo."""
    for name in ("profile.jpg", "profile.jpeg", "profile.png"):
        try:
            return base64.b64encode((APP_DIR / "assets" / name).read_bytes()).decode()
        except OSError:
            continue
    return _EMBEDDED_PHOTO_B64


try:
    model = load_model()
except Exception as exc:
    st.error(
        f"Could not load `{MODEL_PATH.name}`. Place it next to app.py and make sure the "
        f"scikit-learn version matches the one used for training.\n\nDetails: {exc}"
    )
    st.stop()

PHOTO_B64 = load_photo_b64()


def avatar(size: str = "md") -> str:
    """Circular avatar with a thin gradient ring. Falls back to initials."""
    if PHOTO_B64:
        inner = f'<img src="data:image/jpeg;base64,{PHOTO_B64}" alt="Photo of {DEVELOPER}">'
    else:
        inner = f'<div class="ini" role="img" aria-label="{DEVELOPER}">{INITIALS}</div>'
    return f'<div class="avatar {size}">{inner}</div>'


ICONS = {
    "cap": '<path d="M22 10 12 5 2 10l10 5z"/><path d="M6 12v5c3 2 9 2 12 0v-5"/>',
    "zap": '<path d="M13 2 3 14h9l-1 8 10-12h-9z"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "trend": '<path d="M3 17l6-6 4 4 8-8"/><path d="M14 7h7v7"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "book": '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>',
    "moon": '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M8 13h8M8 17h8"/>',
    "ball": '<circle cx="12" cy="12" r="10"/><path d="M12 2a15 15 0 0 1 0 20M12 2a15 15 0 0 0 0 20M2 12h20"/>',
    "layers": '<path d="m12 2 10 5-10 5L2 7z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>',
    "inbox": '<path d="M22 12h-6l-2 3h-4l-2-3H2"/><path d="M5.5 5h13l3.5 7v6a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2v-6z"/>',
}


def icon(name: str, size: int = 18) -> str:
    return (
        f'<svg class="ico" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>'
    )


# ---------------------------------------------------------------
# Styling
# ---------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Sora:wght@500;600;700&display=swap');
:root{
 --ink:#070a18; --surface:#0f1632; --glass:rgba(16,23,51,.66); --surface2:#16204a; --line:#27336b;
 --text:#eaf0ff; --muted:#a3afd3; --accent:#6a7bff; --accent2:#22d3ee; --accent3:#a78bfa;
 --green:#34d399; --amber:#fbbf24; --red:#f87171;
}
html, body, [class*="css"], .stApp { font-family:'Inter',system-ui,sans-serif; }
h1,h2,h3,h4,.display { font-family:'Sora','Inter',sans-serif; }
.stApp {
 background:
  linear-gradient(rgba(255,255,255,.028) 1px, transparent 1px) 0 0 / 46px 46px,
  linear-gradient(90deg, rgba(255,255,255,.028) 1px, transparent 1px) 0 0 / 46px 46px,
  radial-gradient(900px 520px at 88% -8%, rgba(106,123,255,.42), transparent 62%),
  radial-gradient(760px 520px at 2% 12%, rgba(34,211,238,.16), transparent 62%),
  radial-gradient(900px 640px at 62% 112%, rgba(124,92,255,.26), transparent 62%),
  linear-gradient(180deg,#070a18 0%,#05071280 100%), #05070f;
 background-attachment:fixed;
}
header[data-testid="stHeader"] { background:transparent; }
footer { visibility:hidden; }
.block-container { max-width:1180px; padding-top:.8rem; padding-bottom:3rem; }
.ico { flex:none; vertical-align:-3px; }

/* ---------- Avatar ---------- */
.avatar { box-sizing:border-box; flex:none; border-radius:50%; padding:3px;
 background:conic-gradient(from 200deg,var(--accent),var(--accent2),var(--accent)); }
.avatar img, .avatar .ini { width:100%; height:100%; border-radius:50%; display:block; box-sizing:border-box;
 object-fit:cover; object-position:center top; border:3px solid var(--surface); }
.avatar .ini { display:grid; place-items:center; background:var(--surface2); color:#fff; font-weight:700; font-family:'Sora',sans-serif; }
.avatar.xl { width:132px; height:132px; padding:4px; box-shadow:0 18px 44px #6a7bff40; }
.avatar.xl img { border-width:4px; }
.avatar.xl .ini { font-size:2.2rem; }
.avatar.md { width:50px; height:50px; padding:2px; }
.avatar.md img { border-width:2px; }
.avatar.md .ini { font-size:.9rem; }
.avatar.sm { width:34px; height:34px; padding:2px; }
.avatar.sm img { border-width:2px; }

/* ---------- Top bar ---------- */
.nav { display:flex; justify-content:space-between; align-items:center; gap:1rem; flex-wrap:wrap;
 padding:.6rem .7rem .6rem 1rem; background:rgba(18,26,48,.72); border:1px solid var(--line); border-radius:16px;
 backdrop-filter:blur(8px); }
.brand { display:flex; align-items:center; gap:.65rem; color:#fff; font-family:'Sora',sans-serif; font-weight:700; font-size:1.05rem; }
.brand .logo { width:34px; height:34px; border-radius:10px; display:grid; place-items:center; color:#fff;
 background:linear-gradient(135deg,var(--accent),var(--accent2)); }
.brand span.sub { color:var(--muted); font-family:'Inter',sans-serif; font-weight:500; font-size:.82rem; margin-left:.3rem; }
.who { display:flex; align-items:center; gap:.65rem; padding:.2rem 1rem .2rem .25rem; border-radius:999px; background:var(--surface2); border:1px solid var(--line); }
.who b { display:block; color:#fff; font-size:.86rem; line-height:1.15; }
.who small { color:var(--muted); font-size:.72rem; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] { background:var(--glass); backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); border-right:1px solid var(--line); }
section[data-testid="stSidebar"] .stButton > button {
 width:100%; justify-content:flex-start; background:var(--surface2); color:var(--text);
 border:1px solid var(--line); border-radius:10px; padding:.55rem .9rem; font-weight:500;
 transition:border-color .15s ease; }
section[data-testid="stSidebar"] .stButton > button:hover { border-color:var(--accent); color:#fff; }
section[data-testid="stSidebar"] .stButton > button:focus-visible { outline:3px solid var(--amber); outline-offset:2px; }
.side-title { color:var(--muted); font-size:.8rem; font-weight:600; margin:1.2rem 0 .5rem; }
.side-card { background:var(--surface2); border:1px solid var(--line); border-radius:12px;
 padding:.9rem 1rem; color:var(--muted); font-size:.84rem; line-height:1.55; }
.side-card b { color:var(--text); }

/* ---------- Hero ---------- */
.hero { padding:2rem 0 0; }
.kicker { display:inline-flex; align-items:center; gap:.4rem; color:#a9bdff; background:#6a7bff1f; border:1px solid #6a7bff44;
 border-radius:999px; padding:.28rem .8rem; font-size:.8rem; font-weight:600; }
.hero h1 { color:#fff; font-size:clamp(1.9rem,3.8vw,2.65rem); line-height:1.15; font-weight:700; margin:1rem 0 .9rem; padding:0; letter-spacing:-.02em; }
.hero p.lead { color:var(--muted); font-size:1rem; line-height:1.65; max-width:32rem; margin:0 0 1.4rem; }
.facts { display:flex; gap:.6rem; flex-wrap:wrap; margin-bottom:1.6rem; }
.fact { display:flex; align-items:center; gap:.55rem; padding:.55rem .85rem; border:1px solid var(--line); border-radius:12px; background:var(--glass); backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); color:var(--accent2); }
.fact b { color:#fff; font-size:.85rem; display:block; line-height:1.2; }
.fact small { color:var(--muted); font-size:.74rem; }
.tags { display:flex; flex-wrap:wrap; gap:.35rem; }
.tags span { background:#ffffff10; border:1px solid #ffffff1f; color:#d6e0ff; border-radius:999px; padding:.15rem .6rem; font-size:.72rem; font-weight:500; }

/* ---------- Form ---------- */
[data-testid="stForm"] { background:var(--glass); backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); border:1px solid var(--line); border-radius:20px;
 padding:1.5rem 1.5rem 1.1rem; box-shadow:0 24px 60px #00000055; }
.form-head { margin-bottom:.7rem; }
.form-head h3 { margin:0; padding:0; color:#fff; font-size:1.2rem; font-weight:700; }
.form-head p { margin:.2rem 0 0; color:var(--muted); font-size:.86rem; }
[data-testid="stWidgetLabel"] p { color:var(--text); font-weight:500; font-size:.86rem; }
.stFormSubmitButton > button { width:100%; border:none; border-radius:11px; padding:.75rem 1rem; color:#fff;
 font-weight:600; font-size:.98rem; background:var(--accent); transition:filter .15s ease; }
.stFormSubmitButton > button:hover { filter:brightness(1.12); color:#fff; }
.stFormSubmitButton > button:focus-visible { outline:3px solid var(--amber); outline-offset:2px; }

/* ---------- Empty ---------- */
.empty { margin-top:1.8rem; border:1.5px dashed var(--line); border-radius:18px; padding:2.2rem 1.5rem; text-align:center; color:var(--muted); }
.empty .ico { color:var(--accent); margin-bottom:.5rem; }
.empty b { color:var(--text); display:block; font-size:1.02rem; margin-bottom:.3rem; font-family:'Sora',sans-serif; }

/* ---------- Results ---------- */
.res-head { display:flex; justify-content:space-between; align-items:baseline; flex-wrap:wrap; gap:.5rem; margin:2.2rem 0 .4rem; }
.res-head h2 { margin:0; padding:0; color:#fff; font-size:1.35rem; font-weight:700; letter-spacing:-.01em; }
.res-head small { color:var(--muted); font-size:.8rem; }
.stTabs [data-baseweb="tab-list"] { gap:.3rem; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { height:44px; padding:0 1rem; background:transparent; color:var(--muted); font-weight:600; }
.stTabs [aria-selected="true"] { color:#fff; }
.stTabs [data-baseweb="tab-highlight"] { background:var(--accent); }
.stTabs { animation:rise .5s ease backwards; }

.kpis { display:grid; grid-template-columns:1.5fr 1fr 1fr; gap:1rem; margin-top:1.1rem; }
@media (max-width:760px){ .kpis { grid-template-columns:1fr; } }
.kpi { background:var(--glass); backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); border:1px solid var(--line); border-radius:16px; padding:1.1rem 1.25rem; }
.kpi .lbl { display:flex; align-items:center; gap:.4rem; color:var(--muted); font-size:.82rem; font-weight:500; }
.kpi .val { display:block; color:#fff; font-family:'Sora',sans-serif; font-size:2.2rem; font-weight:700; line-height:1.2; margin:.25rem 0 .1rem; letter-spacing:-.02em; }
.kpi .note2 { color:var(--muted); font-size:.8rem; }
.kpi.main { background:linear-gradient(130deg,#2a4bd6,#6a7bff 60%,#22d3ee); border:none; }
.kpi.main .lbl, .kpi.main .note2 { color:#eaf1ff; }
.kpi.main .val { font-size:3rem; }
.up { color:var(--green) !important; } .down { color:var(--red) !important; }
.mini-bar { height:6px; background:#ffffff1c; border-radius:99px; margin-top:.6rem; overflow:hidden; }
.mini-bar div { height:100%; background:var(--accent2); border-radius:99px; }

.grid2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:1rem; margin-top:1rem; }
.card { background:var(--glass); backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); border:1px solid var(--line); border-radius:16px; padding:1.15rem 1.3rem; }
.card h4 { margin:0 0 .2rem; padding:0; color:#fff; font-size:.98rem; font-weight:600; }
.card p.sub { margin:0 0 .6rem; color:var(--muted); font-size:.82rem; }
.card svg.chart { width:100%; height:auto; display:block; }
.gauge-wrap { max-width:300px; margin:0 auto; }
.center { text-align:center; }
.chip { display:inline-block; margin-top:.5rem; padding:.22rem .8rem; border-radius:999px; font-size:.8rem; font-weight:600; border:1px solid currentColor; }

.cmp { margin:.95rem 0; }
.cmp-top { display:flex; justify-content:space-between; align-items:center; color:var(--muted); font-size:.85rem; margin-bottom:.35rem; }
.cmp-top span { display:flex; align-items:center; gap:.4rem; }
.cmp-top b { color:#fff; }
.track { height:10px; border-radius:99px; background:#1d2a4a; overflow:hidden; }
.track div { height:100%; border-radius:99px; }
.verdict { margin-top:1.1rem; padding:.8rem 1rem; border-radius:12px; background:var(--surface2); border:1px solid var(--line); color:var(--text); font-size:.87rem; line-height:1.55; }

.insights { display:grid; gap:.7rem; margin-top:1.1rem; }
.ins { display:flex; align-items:center; gap:1rem; background:var(--glass); backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); border:1px solid var(--line); border-radius:14px; padding:.9rem 1.1rem; }
.ins .rank { width:30px; height:30px; flex:none; border-radius:9px; display:grid; place-items:center; background:#34d3991f; color:var(--green); font-weight:700; font-size:.88rem; }
.ins div.t { flex:1; }
.ins b { color:#fff; display:block; font-size:.93rem; font-weight:600; }
.ins small { color:var(--muted); font-size:.78rem; }
.ins span.g { color:var(--green); font-family:'Sora',sans-serif; font-size:1.35rem; font-weight:700; }
.best { margin-top:1rem; display:flex; justify-content:space-between; align-items:center; gap:1rem; flex-wrap:wrap;
 background:linear-gradient(110deg,#0f766e44,#6a7bff33); border:1px solid #34d39955; border-radius:14px; padding:1rem 1.25rem; }
.best b { color:#fff; font-size:.95rem; }
.best small { color:var(--muted); display:block; font-size:.8rem; }
.best strong { color:var(--green); font-family:'Sora',sans-serif; font-size:1.7rem; font-weight:700; }
.best strong em { font-style:normal; font-size:.85rem; font-family:'Inter',sans-serif; margin-left:.3rem; }
.empty-ins { margin-top:1.1rem; padding:1rem 1.2rem; border-radius:14px; background:var(--glass); backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); border:1px solid var(--line); color:var(--muted); font-size:.9rem; }
.empty-ins b { color:#fff; display:block; margin-bottom:.15rem; }

.note { color:var(--muted); font-size:.8rem; margin-top:1rem; }


/* ===== Theme upgrade layer ===== */
.card, .kpi, .ins, .fact, .nav, .step, [data-testid="stForm"] { box-shadow:inset 0 1px 0 rgba(255,255,255,.07), 0 12px 32px rgba(2,4,16,.38); }
.brand .logo { background:linear-gradient(135deg,var(--accent),var(--accent3)); box-shadow:0 8px 22px rgba(106,123,255,.5); }
.kicker { background:linear-gradient(90deg,#6a7bff26,#22d3ee1a); border-color:#6a7bff55; color:#c3ccff; }
.hero h1 { text-shadow:0 8px 40px rgba(106,123,255,.35); }
.steps { display:grid; grid-template-columns:repeat(3,1fr); gap:.7rem; max-width:34rem; }
@media (max-width:640px){ .steps { grid-template-columns:1fr; } }
.step { padding:.95rem .95rem 1rem; border-radius:14px; background:var(--glass); border:1px solid var(--line); backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); }
.step .n { width:26px; height:26px; border-radius:50%; display:grid; place-items:center; margin-bottom:.6rem; color:#fff;
 font-family:'Sora',sans-serif; font-weight:700; font-size:.78rem; background:linear-gradient(135deg,var(--accent),var(--accent3)); box-shadow:0 6px 16px rgba(106,123,255,.45); }
.step b { display:block; color:#fff; font-size:.88rem; font-weight:600; margin-bottom:.15rem; }
.step small { color:var(--muted); font-size:.76rem; line-height:1.45; display:block; }
.kpi.main { position:relative; overflow:hidden; background:linear-gradient(135deg,#4338ca 0%,#6a7bff 55%,#22d3ee 135%);
 box-shadow:0 20px 48px rgba(106,123,255,.42), inset 0 1px 0 rgba(255,255,255,.28); }
.kpi.main::after { content:""; position:absolute; right:-46px; top:-46px; width:170px; height:170px; border-radius:50%;
 background:radial-gradient(circle, rgba(255,255,255,.30), transparent 65%); pointer-events:none; }
.stFormSubmitButton > button { background:linear-gradient(90deg,#5b6dff,#22a8f0); box-shadow:0 12px 28px rgba(106,123,255,.45); }
.stFormSubmitButton > button:hover { transform:translateY(-1px); }
.stTabs [data-baseweb="tab-list"] { background:var(--glass); border:1px solid var(--line); border-radius:12px; padding:4px; gap:4px; width:fit-content; max-width:100%; }
.stTabs [data-baseweb="tab"] { height:38px; border-radius:9px; padding:0 1.15rem; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,#4f5df0,#6a7bff); color:#fff; box-shadow:0 6px 16px rgba(106,123,255,.4); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
.stSlider [role="slider"] { background:var(--accent); box-shadow:0 0 0 4px rgba(106,123,255,.28); }
.stSlider [data-testid="stThumbValue"] { color:#c9d1ff; font-weight:600; }
.ins .rank { background:linear-gradient(135deg,#34d39933,#22d3ee22); }
.best { box-shadow:0 14px 36px rgba(34,211,238,.14); }

@keyframes rise { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:none; } }
@media (prefers-reduced-motion: reduce) {
 .stTabs { animation:none; }
 section[data-testid="stSidebar"] .stButton > button, .stFormSubmitButton > button { transition:none; }
}
</style>
"""


def html(markup: str):
    """Render HTML; whitespace is collapsed so Markdown never treats it as code."""
    st.markdown(" ".join(markup.split()), unsafe_allow_html=True)


html(CSS)

# ---------------------------------------------------------------
# State + presets
# ---------------------------------------------------------------
DEFAULTS = dict(hours=5, prev=70, extra="Yes", sleep=7, papers=5)
PRESETS = {
    "Top performer": dict(hours=8, prev=92, extra="Yes", sleep=8, papers=8),
    "Balanced": DEFAULTS,
    "Needs support": dict(hours=2, prev=48, extra="No", sleep=5, papers=1),
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_preset(name):
    st.session_state.update(PRESETS[name])


with st.sidebar:
    html('<div class="side-title">Quick profiles</div>')
    for name in PRESETS:
        st.button(name, on_click=apply_preset, args=(name,), key=f"preset_{name}", use_container_width=True)
    html(
        """<div class="side-title">About the model</div>
        <div class="side-card"><b>5 inputs</b> become <b>1 performance index</b> (0 to 100).<br>
        The charts reuse the trained model to test how changing one habit moves the score.</div>"""
    )


# ---------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------
def predict(rows):
    preds = model.predict(pd.DataFrame(rows))
    return [max(0.0, min(100.0, float(p))) for p in preds]


def tier(score):
    if score >= 80:
        return "Excellent", "Outstanding outlook", "#34d399"
    if score >= 60:
        return "Good", "On a solid track", "#22d3ee"
    if score >= 40:
        return "Average", "Room to improve", "#fbbf24"
    return "Needs support", "Needs attention", "#f87171"


def gauge_svg(score, color):
    arc = math.pi * 80
    fill = arc * score / 100
    return f"""<svg class="chart" viewBox="0 0 200 125" role="img" aria-label="Score {score:.1f} out of 100">
    <path d="M20 105 A80 80 0 0 1 180 105" fill="none" stroke="#263356" stroke-width="14" stroke-linecap="round"/>
    <defs><filter id="gl" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
    <path d="M20 105 A80 80 0 0 1 180 105" fill="none" stroke="{color}" stroke-width="14" stroke-linecap="round"
     stroke-dasharray="{fill:.1f} {arc:.1f}" filter="url(#gl)"/>
    <text x="100" y="92" text-anchor="middle" font-size="34" font-weight="700" fill="#fff" font-family="Sora,Inter,sans-serif">{score:.1f}</text>
    <text x="100" y="112" text-anchor="middle" font-size="10" fill="#9aa7c7">out of 100</text></svg>"""


def radar_svg(items):
    cx, cy, r, n = 110, 110, 74, len(items)

    def pt(i, k):
        a = -math.pi / 2 + 2 * math.pi * i / n
        return cx + r * k * math.cos(a), cy + r * k * math.sin(a)

    grid = ""
    for k in (0.25, 0.5, 0.75, 1):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, k) for i in range(n)))
        grid += f'<polygon points="{pts}" fill="none" stroke="#263356"/>'
    axes = "".join(
        f'<line x1="{cx}" y1="{cy}" x2="{pt(i, 1)[0]:.1f}" y2="{pt(i, 1)[1]:.1f}" stroke="#263356"/>' for i in range(n)
    )
    vals = [pt(i, max(v, 0.04)) for i, (_, v) in enumerate(items)]
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in vals)
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="#22d3ee"/>' for x, y in vals)
    labels = ""
    for i, (name, v) in enumerate(items):
        x, y = pt(i, 1.28)
        anchor = "middle" if abs(x - cx) < 6 else ("start" if x > cx else "end")
        labels += (
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="9.5" fill="#9aa7c7">{name} '
            f'<tspan fill="#fff" font-weight="600">{round(v * 100)}</tspan></text>'
        )
    return f"""<svg class="chart" viewBox="-30 -5 280 230" role="img" aria-label="Habit profile radar chart">{grid}{axes}
    <polygon points="{poly}" fill="#6a7bff44" stroke="#98a6ff" stroke-width="2"/>{dots}{labels}</svg>"""


def line_svg(xs, ys, cur, uid):
    W, H, L, B, T, R = 400, 215, 40, 34, 14, 14
    lo, hi = min(ys), max(ys)
    pad = max((hi - lo) * 0.25, 1.0)
    lo, hi = lo - pad, hi + pad
    sx = lambda x: L + (x - xs[0]) / (xs[-1] - xs[0]) * (W - L - R)
    sy = lambda y: T + (hi - y) / (hi - lo) * (H - T - B)
    pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
    area = f"{sx(xs[0]):.1f},{H - B} {pts} {sx(xs[-1]):.1f},{H - B}"
    yt = "".join(
        f'<line x1="{L}" x2="{W - R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="#263356" stroke-dasharray="3 4"/>'
        f'<text x="{L - 8}" y="{sy(v) + 3:.1f}" text-anchor="end" font-size="10" fill="#9aa7c7">{v:.0f}</text>'
        for v in (lo, (lo + hi) / 2, hi)
    )
    xt = "".join(
        f'<text x="{sx(x):.1f}" y="{H - B + 18}" text-anchor="middle" font-size="10" fill="#9aa7c7">{x}</text>' for x in xs
    )
    cy_ = ys[xs.index(cur)]
    return f"""<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="What-if line chart">
    <defs><linearGradient id="g{uid}" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#6a7bff" stop-opacity=".4"/><stop offset="1" stop-color="#6a7bff" stop-opacity="0"/>
    </linearGradient><filter id="f{uid}" x="-10%" y="-10%" width="120%" height="120%"><feGaussianBlur stdDeviation="2.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>{yt}{xt}
    <polygon points="{area}" fill="url(#g{uid})"/>
    <polyline points="{pts}" fill="none" stroke="#98a6ff" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" filter="url(#f{uid})"/>
    <line x1="{sx(cur):.1f}" x2="{sx(cur):.1f}" y1="{T}" y2="{H - B}" stroke="#22d3ee" stroke-dasharray="4 4"/>
    <circle cx="{sx(cur):.1f}" cy="{sy(cy_):.1f}" r="6" fill="#22d3ee" stroke="#070a18" stroke-width="2"/></svg>"""


def compare_html(prev, score, color):
    rows = [("Previous score", prev, "#7c8bb0"), ("Predicted index", score, color)]
    bars = "".join(
        f'<div class="cmp"><div class="cmp-top"><span>{n}</span><b>{v:.1f}</b></div>'
        f'<div class="track" role="img" aria-label="{n} {v:.1f} out of 100"><div style="width:{v:.1f}%;background:{c}"></div></div></div>'
        for n, v, c in rows
    )
    d = score - prev
    if abs(d) < 1:
        msg = "The model expects performance to stay about the same as last time."
    elif d > 0:
        msg = f"The model expects an improvement of <b>{d:.1f} points</b> over the previous score."
    else:
        msg = f"The model expects a drop of <b>{abs(d):.1f} points</b> from the previous score. The action plan shows what could help."
    return f'{bars}<div class="verdict">{msg}</div>'


def habit_row(ic, name, value, pct, color):
    return (
        f'<div class="cmp"><div class="cmp-top"><span>{icon(ic, 16)}{name}</span><b>{value}</b></div>'
        f'<div class="track"><div style="width:{pct:.0f}%;background:{color}"></div></div></div>'
    )


# ---------------------------------------------------------------
# Top bar
# ---------------------------------------------------------------
html(
    f"""<div class="nav"><div class="brand"><div class="logo">{icon("cap", 19)}</div>
    EduPredict<span class="sub">Student performance</span></div>
    <div class="who">{avatar("md")}<div><b>{DEVELOPER}</b><small>Developer</small></div></div></div>"""
)

# ---------------------------------------------------------------
# Hero + form
# ---------------------------------------------------------------
left, right = st.columns([1.15, 1], gap="large")

with left:
    html(
        f"""<div class="hero"><span class="kicker">{icon("zap", 15)} Trained ML model</span>
        <h1>Forecast a student's results, and see what moves them.</h1>
        <p class="lead">Enter study habits and past scores. The model returns a performance index and ranks
        the habit changes that would raise it the most.</p>
        <div class="facts">
        <div class="fact">{icon("layers")}<div><b>5 inputs</b><small>habits and scores</small></div></div>
        <div class="fact">{icon("target")}<div><b>0 to 100</b><small>performance index</small></div></div>
        <div class="fact">{icon("trend")}<div><b>What-if</b><small>ranked by impact</small></div></div></div>
        <div class="steps">
        <div class="step"><div class="n">1</div><b>Enter the profile</b><small>Study hours, sleep, practice and past scores</small></div>
        <div class="step"><div class="n">2</div><b>Get the forecast</b><small>The model returns a 0 to 100 index</small></div>
        <div class="step"><div class="n">3</div><b>Follow the plan</b><small>Ranked changes with estimated gains</small></div></div></div>"""
    )

with right:
    with st.form("prediction_form"):
        html(
            """<div class="form-head"><h3>Student profile</h3>
            <p>Enter the details to get a prediction</p></div>"""
        )
        c1, c2 = st.columns(2)
        with c1:
            hours = st.slider("Hours studied per day", 1, 9, key="hours")
            papers = st.slider("Sample papers practiced", 0, 9, key="papers")
        with c2:
            sleep_hours = st.slider("Sleep hours per night", 4, 9, key="sleep")
            previous_scores = st.slider("Previous scores", 40, 99, key="prev")
        extracurricular = st.radio("Extracurricular activities", ["Yes", "No"], horizontal=True, key="extra")
        submit = st.form_submit_button("Predict performance")

# ---------------------------------------------------------------
# Run the model
# ---------------------------------------------------------------
if submit:
    profile = {
        "Hours Studied": hours,
        "Previous Scores": previous_scores,
        "Extracurricular Activities": extracurricular,
        "Sleep Hours": sleep_hours,
        "Sample Question Papers Practiced": papers,
    }
    try:
        with st.spinner("Calculating..."):
            score = predict([profile])[0]
            xs_h, xs_p = list(range(1, 10)), list(range(0, 10))
            ys_h = predict([{**profile, "Hours Studied": h} for h in xs_h])
            ys_p = predict([{**profile, "Sample Question Papers Practiced": p} for p in xs_p])

            levers = []
            if hours < 9:
                levers.append(("Study 1 more hour a day", {"Hours Studied": hours + 1}))
            if papers < 9:
                n = min(9, papers + 2)
                levers.append((f"Practice {n - papers} more sample papers", {"Sample Question Papers Practiced": n}))
            if sleep_hours < 9:
                levers.append(("Sleep 1 more hour", {"Sleep Hours": sleep_hours + 1}))
            if extracurricular == "No":
                levers.append(("Join an extracurricular activity", {"Extracurricular Activities": "Yes"}))

            gains = []
            if levers:
                preds = predict([{**profile, **c} for _, c in levers])
                gains = [(lab, p - score, c) for (lab, c), p in zip(levers, preds)]
            gains = sorted([g for g in gains if g[1] > 0.05], key=lambda g: -g[1])[:3]

            best = None
            if gains:
                merged = {}
                for _, _, c in gains:
                    merged.update(c)
                best = predict([{**profile, **merged}])[0]
    except Exception as exc:
        st.error(f"The model could not make a prediction. Check that the input columns match training.\n\nDetails: {exc}")
        st.stop()

    st.session_state["result"] = dict(
        profile=profile, score=score, xs_h=xs_h, ys_h=ys_h, xs_p=xs_p, ys_p=ys_p,
        gains=[(lab, g) for lab, g, _ in gains], best=best,
        stamp=datetime.now().strftime("%d %b %Y, %H:%M"),
    )

# ---------------------------------------------------------------
# Results
# ---------------------------------------------------------------
res = st.session_state.get("result")

if not res:
    html(
        f"""<div class="empty">{icon("inbox", 30)}<b>Your results will appear here</b>
        Fill in the student profile and select Predict performance, or start from a quick profile in the sidebar.</div>"""
    )
else:
    p, score = res["profile"], res["score"]
    label, blurb, color = tier(score)
    h, sl, pa = p["Hours Studied"], p["Sleep Hours"], p["Sample Question Papers Practiced"]
    prev, extra = p["Previous Scores"], p["Extracurricular Activities"]

    radar_items = [
        ("Study", h / 9),
        ("Sleep", (sl - 4) / 5),
        ("Practice", pa / 9),
        ("Previous", (prev - 40) / 59),
        ("Activities", 1.0 if extra == "Yes" else 0.4),
    ]
    habit = round(sum(v for n, v in radar_items if n != "Previous") / 4 * 100)
    delta = score - prev
    arrow, dcls = ("▲", "up") if delta >= 0 else ("▼", "down")

    html(f'<div class="res-head"><h2>Prediction results</h2><small>Generated {res["stamp"]}</small></div>')
    tab1, tab2, tab3 = st.tabs(["Overview", "What-if analysis", "Action plan"])

    with tab1:
        html(
            f"""<div class="kpis">
            <div class="kpi main"><span class="lbl">{icon("target", 16)} Predicted index</span>
            <span class="val">{score:.1f}</span><span class="note2">{label} · {blurb}</span></div>
            <div class="kpi"><span class="lbl">{icon("trend", 16)} Change vs previous</span>
            <span class="val {dcls}">{arrow} {abs(delta):.1f}</span><span class="note2">Previous score {prev}</span></div>
            <div class="kpi"><span class="lbl">{icon("activity", 16)} Habit score</span>
            <span class="val">{habit}%</span><div class="mini-bar"><div style="width:{habit}%"></div></div></div></div>"""
        )
        html(
            f"""<div class="grid2">
            <div class="card"><h4>Performance gauge</h4><p class="sub">Predicted index on a 0 to 100 scale</p>
            <div class="gauge-wrap">{gauge_svg(score, color)}</div>
            <div class="center"><span class="chip" style="color:{color}">{label}</span></div></div>
            <div class="card"><h4>Habit profile</h4><p class="sub">Each axis is scaled 0 to 100 within the allowed range</p>
            {radar_svg(radar_items)}</div></div>"""
        )
        html(
            f"""<div class="grid2">
            <div class="card"><h4>Previous vs predicted</h4><p class="sub">How the forecast compares with past results</p>
            {compare_html(prev, score, color)}</div>
            <div class="card"><h4>Habit snapshot</h4><p class="sub">The inputs behind this prediction</p>
            {habit_row("book", "Study hours per day", h, h / 9 * 100, "#6a7bff")}
            {habit_row("moon", "Sleep hours per night", sl, (sl - 4) / 5 * 100, "#22d3ee")}
            {habit_row("file", "Sample papers practiced", pa, pa / 9 * 100, "#34d399")}
            {habit_row("ball", "Extracurricular activities", extra, 100 if extra == "Yes" else 0, "#fbbf24")}</div></div>"""
        )

    with tab2:
        html(
            f"""<div class="grid2">
            <div class="card"><h4>Hours studied vs. predicted index</h4><p class="sub">Other inputs held constant. Marker shows the current value.</p>
            {line_svg(res["xs_h"], res["ys_h"], h, "h")}</div>
            <div class="card"><h4>Sample papers vs. predicted index</h4><p class="sub">Other inputs held constant. Marker shows the current value.</p>
            {line_svg(res["xs_p"], res["ys_p"], pa, "p")}</div></div>"""
        )

    with tab3:
        if res["gains"]:
            rows = "".join(
                f'<div class="ins"><div class="rank">{i}</div><div class="t"><b>{lab}</b>'
                f'<small>estimated points on the index</small></div><span class="g">+{g:.1f}</span></div>'
                for i, (lab, g) in enumerate(res["gains"], 1)
            )
            html(f'<div class="insights">{rows}</div>')
        else:
            html(
                """<div class="empty-ins"><b>These habits are already near the model's best</b>
                No single change is predicted to raise the index.</div>"""
            )
        if res.get("best") is not None:
            html(
                f"""<div class="best"><div><b>If all of these changes are made together</b>
                <small>Combined estimate from the trained model</small></div>
                <strong>{res["best"]:.1f}<em>(+{res["best"] - score:.1f})</em></strong></div>"""
            )
        report = pd.DataFrame([{**p, "Predicted Index": round(score, 1), "Level": label, "Generated": res["stamp"]}])
        st.download_button(
            "Download report (CSV)",
            report.to_csv(index=False).encode("utf-8"),
            file_name="student_prediction_report.csv",
            mime="text/csv",
        )
        html('<div class="note">Estimates come from your trained machine learning model and are not guaranteed results.</div>')