import json

failed_out = "Raw-news-sinhala-failed.txt"
failed_list = "failed.list"

ff = open(failed_out, "w")
failed_files = open(failed_list).readlines()
for name in failed_files:
    with open(name.strip()) as fl:
        lines = fl.readlines()
        if len(lines) > 1:
            print("More than 1 line")
        lines = lines[0]
        lines = lines.split("}{")
        try:
            for j in range(len(lines)):
                print(lines[j], j)
                if j % 2 == 0:
                    d = json.loads(lines[j] + "}")
                else:
                    d = json.loads("{" + lines[j])
                for i in d['Content']:
                    ff.write(i + "\n")
        except Exception as e:
            print(name, e)
ff.close()
