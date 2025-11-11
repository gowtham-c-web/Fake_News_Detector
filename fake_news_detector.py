import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


fake_csv_path = r"C:\Users\Dell\Desktop\DEV\Fake.csv"
true_csv_path = r"C:\Users\Dell\Desktop\DEV\True.csv"

try:
    fake_df = pd.read_csv(fake_csv_path)
    true_df = pd.read_csv(true_csv_path)
except FileNotFoundError as e:
    print(f"Error: CSV file not found. {e}")
    print("Please check the file paths and try again.")
    exit()

fake_df["label"] = "FAKE"
true_df["label"] = "REAL"

df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("✅ Data loaded successfully!")
print(f"Total REAL articles: {len(true_df)}")
print(f"Total FAKE articles: {len(fake_df)}")


if "text" in df.columns:
    feature_column = "text"
elif "title" in df.columns:
    feature_column = "title"
else:
    raise ValueError("No 'text' or 'title' column found in the dataset!")

def clean_text(text):
    if not isinstance(text, str):
        return "" 
        
    text = re.sub(r'^[A-Z\s]+\s*\([A-Za-z]+\s*\)\s*-\s*', '', text)
    text = re.sub(r'^\(Reuters\)\s*-\s*', '', text)
    return text

df[feature_column] = df[feature_column].apply(clean_text)
# ------------------------------------

df = df.dropna(subset=[feature_column])
df = df[df[feature_column] != ""] 
X = df[feature_column] 
y = df["label"]

if X.empty:
    raise ValueError("No data left after cleaning. Check your CSV files.")

print(f"Total samples: {len(X)}")
print(df.head())

# Split into training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf.fit_transform(X_train)
tfidf_test = tfidf.transform(X_test)

model = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
model.fit(tfidf_train, y_train)
print("\n✅ Model trained successfully!")

y_pred = model.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {score*100:.2f}%")

print("\n--- Example Predictions from Test Set ---")
results_df = pd.DataFrame({
    'Actual Label': y_test, 
    'Predicted Label': y_pred, 
    'Article Text': X_test
})
print(results_df.head(10))

while True:
    user_input = input("\n📰 Enter a news headline/article (or type 'exit' to quit):\n> ")
    if user_input.lower() == "exit":
        break
    
    if not user_input.strip():
        print("Please enter some text.")
        continue
    
    cleaned_input = clean_text(user_input)
    user_tfidf = tfidf.transform([cleaned_input])
    prediction = model.predict(user_tfidf)[0]
    print(f"🔍 This news is predicted to be: **{prediction}**")
