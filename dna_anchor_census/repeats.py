def repeats(s):
    marked = [False] * len(s)
    for period in range(1, 7):
        start = None
        for i in range(period, len(s) + 1):
            match = i < len(s) and s[i] == s[i-period]
            if match and start is None:
                start = i-period
            if not match and start is not None:
                if i-start >= max(12, 3*period):
                    marked[start:i] = [True] * (i-start)
                start = None
    return [sum(marked[j:j+6]) >= 3 for j in range(0, 2046, 6)]
