# CHAT BOT
# PACKAGES
import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

#Intents

intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hi there", "Hello", "Hey", "I'm fine, thank you", "Nothing much"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?",
                      "How can I assist you?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age"],
        "responses": ["I don't have an age. I'm a chatbot.", "I was just born in the digital world.",
                      "Age is just a number for me."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": ["I'm sorry, I cannot provide real-time weather information.",
                      "You can check the weather on a weather app or website."]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
        "responses": [
            "To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent, food, and bills. Next, allocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.",
            "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.",
            "To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses."]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
        "responses": [
            "A credit score is a number that represents your creditworthiness. It is based on your credit history and is used by lenders to determine whether or not to lend you money. The higher your credit score, the more likely you are to be approved for credit.",
            "You can check your credit score for free on several websites such as Credit Karma and Credit Sesame."]
    },
    {
        "tag": "Python_description",
        "patterns": ["What is Python", "Python", "Who discovered python", "Python language", "Use of Python"],
        "responses": [
            "Python is an interpreted, object-oriented, high-level programming language with dynamic semantics. Its high-level built in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development, as well as for use as a scripting or glue language to connect existing components together.",
            " Python's simple, easy to learn syntax emphasizes readability and therefore reduces the cost of program maintenance. Python supports modules and packages, which encourages program modularity and code reuse. The Python interpreter and the extensive standard library are available in source or binary form without charge for all major platforms, and can be freely distributed."]
    },
    {
        "tag": "Python_packages",
        "patterns": ["Packages", "Python Packages", "Types of packages", "Type of Python Packages",
                     "Uses of Python Packages"],
        "responses": [
            "The Different Types of packages are: 1.Numpy\n 2.Pandas\n 3.Matplotlib\n 4.Seaborn\n 5.Scikit-learn\n 6.Requests\n 7.NLTK\n 8.Pytest\n 9.Pillow"]
    },
    {
        "tag": "Python_numpy_description",
        "patterns": ["Numpy Package", "Python Numpy", "Use of Numpy", "Numpy", "Numpy tool"],
        "responses": [
            "NumPy is the primary tool for scientific computing in Python. It combines the flexibility and simplicity of Python with the speed of languages like C and Fortran.",
            "NumPy is used for:1.Advanced array operations (e.g. add, multiply, slice, reshape, index).\n 2.Comprehensive mathematical functions.\n 3.Random number generation.\n 4.Linear algebra routines.\n 5.Fourier transforms, etc.",
            "With NumPy, you are getting the computational power of compiled code, while using accessible Python syntax. No wonder that there is a huge ecosystem of Python packages and libraries drawing on the power of NumPy. These include such popular packages as pandas, Seaborn, SciPy, OpenCV, and others."]
    },
    {
        "tag": "Python_pandas_description",
        "patterns": ["pandas Package", "Python pandas", "Use of pandas", "pandas", "pandas tool"],
        "responses": [
            "Pandas is known as a fast, efficient, and easy-to-use tool for data analysis and manipulation. It works with data frame objects; a data frame is a dedicated structure for two-dimensional data. Data frames have rows and columns just like database tables or Excel spreadsheets.",
            "pandas can be used for:1.Reading/writing data from/to CSV and Excel files and SQL databases.\n2.Reshaping and pivoting datasets.\n3.Slicing, indexing, and subsetting datasets.\n4.Aggregating and transforming data.\n5.Merging and joining datasets.\n"]
    },
    {
        "tag": "Python_matplotlib_description",
        "patterns": ["matplotlib Package", "Python matplotlib", "Use of matplotlib", "matplotlib", "matplotlib tool"],
        "responses": [
            "Matplotlib is the most common data exploration and visualization library. You can use it to create basic graphs like line plots, histograms, scatter plots, bar charts, and pie charts. You can also create animated and interactive visualizations with this library. Matplotlib is the foundation of every other visualization library.",
            "Matplootlib library offers a great deal of flexibility with regards to formatting and styling plots. You can freely choose how to display labels, grids, legends, etc. However, to create complex and visually appealing plots, you'll need to write quite a lot of code."]
    },
    {
        "tag": "Python_seaborn_description",
        "patterns": ["seaborn Package", "Python seaborn", "Use of seaborn", "seaborn", "seaborn tool"],
        "responses": [
            "Seaborn is a high-level interface for drawing attractive statistical graphics with just a few lines of code."]
    },
    {
        "tag": "Python_scikit_description",
        "patterns": ["Scikit Package", "Python scikit", "Use of scikit", "scikit", "scikit tool"],
        "responses": ["Scikit-learn is an efficient and beginner-friendly tool for predictive data analysis.",
                      "Scikit-learn makes machine learning with Python accessible to people with minimal programming experience. With just a few lines of code, you can model your data using algorithms like random forest, support vector machines (SVM), k-means, spectral clustering, and more."]

    },
    {
        "tag": "Python_nltk_description",
        "patterns": ["nltk Package", "Python nltk", "Use of nltk", "nltk", "nltk tool"],
        "responses": [
            "Natural Language Toolkit (NLTK) is one of the leading Python platforms for processing language data.",
            "Nltk is a set of language processing libraries and programs that provide a toolkit for:1.Classification.\n2.Tokenization.\n3.Stemming.\n4.Tagging.\n5.Parsing.\n6.Semantic reasoning."]
    },
]

#Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

#Chat Bot function

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

#Main Function

counter = 0

def main():
    global counter
    st.title("Chatbot")
    st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

    counter += 1
    user_input = st.text_input("You:", key=f"user_input_{counter}")

    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None, key=f"chatbot_response_{counter}")

        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

if __name__ == '__main__':
    main()