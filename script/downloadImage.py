from pymongo import MongoClient
import requests as req
from io import BytesIO
from PIL import Image
import time
import argparse
global client, painting_collection


def readTxt(filename,model="r"):
    fr = open(filename, "a")
    fr.close()
    fr = open(filename,model)
    data = []
    for line in fr.readlines():
        line = line.strip(" ").split("\t")
        data.append(line[0])
    fr.close()
    return data

def wrateTotxt(filename,data,model = "a"):
    fr = open(filename, model)
    for i in range(len(data) - 1):
        fr.write(str(data[i]))
        fr.write("\t")
    fr.write(str(data[-1]))
    fr.write("\n")
    fr.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-begin',
        type=int,
        default=1,
        help='Model to be validated'
    )
    parser.add_argument(
        '-end',
        type=int,
        default=5,
        help='Model to be validated'
    )
    args = parser.parse_args()
    start_num = args.begin
    end_num = args.end

    #线上环境
    #client = MongoClient('192.168.1.5', 27017)
    #client = MongoClient('115.231.226.46', 27017)
    #db_auth = client.dlsdata_v2
    #db_auth.authenticate("pgc", "GPEGssLpHP04")

    #测试环境
    client = MongoClient('10.4.40.129', 27017)
    db = client.dlsdata_v2

    painting_collection = db.painting

    paint_cursor = painting_collection.find({}, no_cursor_timeout=True)

    ind = 0
    for paint in paint_cursor:
        files = readTxt("../images/data.txt")
        ind += 1
        if paint["image"] in files:
            continue
        if start_num <= ind <= end_num:

            try:
                s = time.time()
                original_path = "https://pic.allhistory.com/" + paint["image"]
                #print(original_path)
                response = req.get(original_path)
                if int(response.status_code) == 200:
                    im = Image.open(BytesIO(response.content))
                    # im=im.resize((672,672))
                    resize_path = "../images/"+paint["image"]
                    im.save(resize_path)
                    data = []
                    data.append(paint["image"])
                    data.append(paint['_id'])

                    wrateTotxt("../images/data.txt",data)

                    print("process time %d: %s" % (ind, time.time() - s))
                # break
            except Exception:
                print("error", paint["image"])
                continue
        else:
            print("has to end")
            break

    paint_cursor.close()


