"""Look at the first lines of a data file (Windows has no 'head').   python peek.py path/to/file [n_lines]"""
import sys, os, collections
p = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
print("file:", p, "| size: %.1f MB" % (os.path.getsize(p) / 1e6))
with open(p, "rb") as f:
    head = f.read(2048)
if b"\x00" in head or head[:2] in (b"\x80\x04", b"\x80\x03", b"\x80\x02") or head[:6] == b"\x93NUMPY" or head[:2] == b"PK":
    print("BINARY file (pickle/npz/zip/mat?) - this toolkit reads TEXT files. Paste the output of `dir` for this folder instead.")
    sys.exit()
widths = collections.Counter(); lines = 0
with open(p, errors="replace") as f:
    for i, line in enumerate(f):
        lines += 1
        if i < n: print(repr(line[:160]))
        if i < 2000: widths[len(line.split())] += 1
print("total lines:", lines)
print("columns per line (first 2000 lines):", dict(widths.most_common(6)))
if len(widths) > 3: print("-> looks like an ADJACENCY LIST ('user item1 item2 ...'): use --format adjlist for the interaction file")
elif widths: print("-> looks like %d columns per line: use --inter-cols / --social-cols to choose the two id columns (0-based)" % widths.most_common(1)[0][0])
