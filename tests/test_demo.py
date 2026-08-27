from asl.infer import apply_letter


def test_apply_letter_spells_and_deletes():
    buf: list[str] = []
    apply_letter(buf, "H")
    apply_letter(buf, "I")
    apply_letter(buf, "SPACE")
    apply_letter(buf, "A")
    apply_letter(buf, "DELETE")
    assert "".join(buf) == "HI "
