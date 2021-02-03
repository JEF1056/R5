import io

with io.open("file", mode="r", encoding="utf-8") as f:
    for i in range(8):
        d=f.readline()
        if d == "":
            f.seek(0)
            d=f.readline()
        print(d.replace("\n",""))