def dec_to_ternary(n):
    if n == 0:
        return "0"
    out = ""
    while n:
        out += str(n%3)
        n //= 3
    return out[::-1]

def ternary_to_balanced_ternary(ternary):
    out = ""
    carry = 0
    for ch in ternary[::-1]:
        if ch == "0":
            out += "0" if not carry else "1"
            carry = 0
        elif ch == "1":
            if carry:
                out += "Z"
            else:
                out += "1"
        else:
            if carry:
                out += "1"
            else:
                out += "Z"
                carry = 1
    if carry:
        out += "1"
    return out[::-1]

def dec_to_balanced_ternary(n):
    ternary = dec_to_ternary(n)
    return ternary_to_balanced_ternary(ternary)

for i in range(11):
    ans = dec_to_balanced_ternary(i)
    print(i, ans)