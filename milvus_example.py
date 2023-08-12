from arcface.ArcFace import ArcFace
import cv2
from retinaface import RetinaFace
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)
import numpy as np

face_rec = ArcFace()
# Connect to Milvus server
def create_connection():
    print("Create connection...")
    connections.connect(host='localhost', port='19530',alias='default')

# Create a collection named 'Image'
def create_collection():
    collection_name = 'Image'
    id_field = 'id'
    vector_field = 'embedding'

    field_id = FieldSchema(name=id_field, dtype=DataType.INT64, is_primary=True)
    field_embedding = FieldSchema(name=vector_field, dtype=DataType.FLOAT_VECTOR, dim=512)

    schema = CollectionSchema(fields=[field_id, field_embedding], description="Image collection")
    collection = Collection(name=collection_name, schema=schema)
    print("Collection created:", collection_name)
    return collection

# Insert data into collection
def insert_data(collection, num, face_embedding):
    data = [
        [num],
        [face_embedding],
    ]

    collection.insert(data)
    return data[1]
           


# Create index for the vector field
def create_index(collection):
    index_param = {
        'index_type': 'IVF_FLAT',
        'params': {'nlist': 1024},
        'metric_type': 'L2'
    }
    collection.create_index(field_name='embedding', index_params=index_param)
    print("Index created for the vector field.")

# Search for similar faces
def search_faces(collection, search_vectors):
    

    search_param = {
        'data': [search_vectors],
        'anns_field': 'embedding',
        'param': {'metric_type': 'L2'},
        'limit': 5
    }

    results = collection.search(**search_param)
    print("Search Results:")
    for res in results:
        if res:
            print(res[0])
        # for r in res:
        #     print(r)
        


# Main function
def main():
    # Connect to Milvus server
    create_connection()

    # Create a collection
    collection = create_collection()

    # Insert data
    img_paths = ["data/img11.jpg", "data/img14.jpg"]

    for index, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        faces = RetinaFace.extract_faces(img, align=True)
        print('face',faces)

        face_embedding = face_rec.calc_emb(faces)
        print('len',len(face_embedding))
        # insert_data(collection, index, face_embedding[0].tolist())

            
    

    # Create index
    create_index(collection)

    # Search for similar faces
    img_pathes = "outputs/img11.jpg"
    img = cv2.imread(img_path)
    faces = RetinaFace.extract_faces(img, align=True)

    face_embeddings = face_rec.calc_emb(faces)
    vector_search = face_embedding[0].tolist()
    search_faces(collection, vector_search)

    # Disconnect from Milvus server
    # connections.disconnect()

if __name__ == '__main__':
    main()
