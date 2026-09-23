import re

src = open("NOTES.md").read()
# extract the inserted section 10 block (from "## 10." up to just before "Files: leaderboard/regression_2026-09-23")
start = src.index("## 10. POST-REVEAL ITERATION LOG")
end = src.index("Files: leaderboard/regression_2026-09-23")
block = src[start:end]
rest = src[:start] + src[end:]
assert "## 10." not in rest
rest = rest.rstrip() + "\n\n" + block.rstrip() + "\n"
open("NOTES.md", "w").write(rest)
print("moved; new order:")
for m in re.finditer(r"^## .*$", rest, re.M):
    print(m.group(0))
