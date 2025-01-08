import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


file_path = '/kaggle/input/edgeiiotset-cyber-security-dataset-of-iot-iiot/Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv'
df = pd.read_csv(file_path , low_memory = False)
df = shuffle(df)
df = df.sample(frac = 0.25, random_state=42)
# Dropping unneseccary data

drop_columns = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4","arp.dst.proto_ipv4", "http.file_data","http.request.full_uri","icmp.transmit_timestamp","http.request.uri.query", "tcp.options","tcp.payload","tcp.srcport","tcp.dstport", "udp.port", "mqtt.msg"]

df.drop(drop_columns, axis=1, inplace=True)
df.dropna(axis=0, how='any', inplace=True)
df.drop_duplicates(subset=None, keep="first", inplace=True)
df.isna().sum()

#Encoding text data

def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)

encode_text_dummy(df,'http.request.method')
encode_text_dummy(df,'http.referer')
encode_text_dummy(df,"http.request.version")
encode_text_dummy(df,"dns.qry.name.len")
encode_text_dummy(df,"mqtt.conack.flags")
encode_text_dummy(df,"mqtt.protoname")
encode_text_dummy(df,"mqtt.topic")

df.to_csv('preprocessed_DNN.csv', encoding='utf-8')

from sentence_transformers import SentenceTransformer
import faiss

data_chunks = ["DDoS_UDP: Overloads a target with high-volume UDP packets to exhaust its resources.",
    "DDoS_ICMP: Uses ping floods to deplete a target's bandwidth and processing power.",
    "DDoS_TCP: Exploits TCP handshakes with excessive SYN packets to drain server resources.",
    "DDoS_HTTP: Overwhelms a server with HTTP requests, disrupting web services.",
    "MITM: Intercepts and alters communication between two parties to steal sensitive data.",
    "Fingerprint: Collects system or application details for exploitation through probing.",
    "SQL_Injection: Injects malicious SQL commands to access or manipulate databases.",
    "File_Upload: Introduces harmful files or executes unauthorized code via uploads.",
    "Password_Attack: Gains unauthorized access by cracking or guessing passwords.",
    "Backdoor: Installs hidden access points to bypass security measures covertly.",
    "Port_Scan: Detects open ports for potential attacks by probing systems.",
    "Vulnerability_Scan: Identifies system weaknesses for potential exploitation.",
    "XSS: Embeds malicious scripts in web pages to steal data or hijack sessions.",
    "Ransomware: Encrypts files, demanding payment for decryption.",
    "Normal: Legitimate and safe network traffic without malicious activity."]

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for data chunks
embeddings = model.encode(data_chunks)

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

def retrieve_context(query, model, index, data_chunks, top_k=1):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=top_k)
    return [data_chunks[i] for i in I[0]]

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score

df = pd.read_csv('preprocessed_DNN.csv' , low_memory = False)

#Encode Labels

labels = df['Attack_type']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

#Extracting Features

features = df.drop(columns = ["Attack_label","Attack_type"],inplace = True)

# Augmenting

def augment_features(features, context, model):
    context_embedding = model.encode([" ".join(context)]).flatten()
    return features + context_embedding.tolist()
    
# Augment features with context

augmented_features = []
for i, row in df.iterrows():
    query = labels.iloc[i]
    context = retrieve_context(query, model, index, data_chunks)
    augmented_features.append(augment_features(row.tolist(), context, model))
    

augmented_features = np.array(augmented_features)

#Splitting data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(augmented_features, labels, test_size=0.2, random_state=42)

# Training random forest classifier

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# Applying 5-fold cross-validation on the training data
cv_scores = cross_val_score(rf_classifier, X_train, y_train, cv=5)

# Output the cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())
