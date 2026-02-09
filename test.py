import pickle

with open("label_processor.pkl", "rb") as f:
    obj = pickle.load(f)

print(type(obj))
print(obj.keys())
