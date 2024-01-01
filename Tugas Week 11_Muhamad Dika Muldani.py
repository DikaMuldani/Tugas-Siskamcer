Python 3.11.2 (tags/v3.11.2:878ead1, Feb  7 2023, 16:38:35) [MSC v.1934 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import pandas as pd
... from sklearn.model_selection import train_test_split
... from sklearn.feature_extraction.text import CountVectorizer
... from sklearn.svm import SVC
... from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
... 
... # Contoh dataset
... data = {
...     'byte_sequence': [
...         '0x5a4d4f53', '0x5a4d4f53', '0x4d5a9000', '0x4d5a9000', '0x5a4d4f53',
...         '0x4d5a9000', '0x5a4d4f53', '0x4d5a9000', '0x5a4d4f53', '0x5a4d4f53',
...         '0x4d5a9000', '0x5a4d4f53', '0x4d5a9000', '0x5a4d4f53', '0x5a4d4f53',
...         '0x4d5a9000', '0x5a4d4f53', '0x4d5a9000', '0x5a4d4f53', '0x4d5a9000'
...     ],
...     'label': [1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
... }
... 
... # Buat DataFrame
... df = pd.DataFrame(list(zip(data['byte_sequence'], data['label'])), columns=['byte_sequence', 'label'])
... 
... # Ekstraksi fitur dari byte_sequence (contoh sederhana menggunakan CountVectorizer)
... vectorizer = CountVectorizer()
... X = vectorizer.fit_transform(df['byte_sequence'])
... 
... # Membagi dataset
... X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)
... 
... # Inisialisasi model SVM
... svm_model = SVC(kernel='linear', C=1)
... 
... # Melatih model SVM
... svm_model.fit(X_train, y_train)
... 
# Memprediksi kelas untuk set pengujian
y_pred = svm_model.predict(X_test)

# Mengukur akurasi model
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi Model SVM: {accuracy}')

# Menampilkan evaluasi model yang lebih komprehensif
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Menampilkan confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
