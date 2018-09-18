from pymongo import MongoClient
from elasticsearch import Elasticsearch


if __name__ == "__main__":
    print("start process")
    es = Elasticsearch(['115.231.226.151:10200'])
    #es = Elasticsearch(['10.4.40.181:9200'])

    #es.indices.create(index='painting_search_cnn_new', ignore = 400)

    client = MongoClient('115.231.226.46', 27017)
    db_auth = client.dlsdata_v2
    db_auth.authenticate("pgc", "GPEGssLpHP04")

    #client = MongoClient('10.4.40.129', 27017)

    db = client.dlsdata_v2
    #con_paint_feature_cnn = db.painting_cnn_features
    con_paint_feature_cnn = db.painting_cnn_new_features
    con_paint_feature = db.painting
    paint_cursor = con_paint_feature_cnn.find({})
    print("db connect sucess")
    count = 0
    for paint_cnn in paint_cursor:
        count += 1
        if count % 100 == 0:
            print(count)
        query = {}
        query['image'] = paint_cnn['imageName']
        paint = con_paint_feature.find_one(query)
        if not paint:
            continue
        data = {}
        metadata = {}
        metadata['width'] = paint['width']
        metadata['height'] = paint['height']
        data['metadata'] = metadata
        if "itemId" in paint.keys():
            data['itemId'] = paint['itemId']
        else:
            data['itemId'] = ""
        data['title'] = paint['title']
        data['scope'] = "paint"
        url = "https://pic.allhistory.com/"+paint['image']
        data['url'] = url
        data['id'] = paint_cnn['imageId'].strip('\n')
        data['signature'] = paint_cnn['features']
        for i in range(len(paint_cnn['mapFeatures'])):
            strname = "simple_word_" + str(i)
            data[strname] = int(paint_cnn['mapFeatures'][i])
        es.index(index="painting_search_cnn_new", doc_type="painting_search_cnn_new", body=data)









