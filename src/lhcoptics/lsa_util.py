_lsa = None

def get_lsa():
    global _lsa
    if _lsa is None:
        import pjlsa

        _lsa = pjlsa.LSAClient()
    return _lsa
