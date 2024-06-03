
<img width="853" alt="image" src="https://github.com/bcullu/PPAD_24/assets/82031922/b5c7a9f0-b9a3-43d5-b557-1fbb2d0db25c">




# Chat with Cruise Passengers

This project is a Streamlit web application that allows users to chat with cruise passengers by querying reviews of specific ships. The application uses LangChain, OpenAI, and FAISS for conversational retrieval and embedding.

## Features

- **Interactive Chat Interface:** Chat with cruise passengers and get responses based on consumer reviews.
- **Dynamic Vector Store:** Efficiently retrieve relevant reviews using a vector store with FAISS.
- **Conversational Memory:** Maintain conversation context using a conversational retrieval chain.

## Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/bcullu/PPAD_24.git
   cd PPAD_24
   ```

2. **Create a virtual environment:**

   ```sh
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install the required packages:**

   ```sh
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**

   Create a `.env` file in the root directory and add your OpenAI API key:

   ```sh
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

1. **Run the Streamlit app:**

   ```sh
   streamlit run app.py
   ```

2. **Interact with the app:**

   - Enter the name of a ship (Celestyal Crystal or Celestyal Olympia) to fetch reviews.
   - Ask your questions to the passengers based on the reviews.

## Code Explanation

- **Dependencies:**
  - `streamlit`: For creating the web application.
  - `pandas`: For data manipulation.
  - `langchain`: For text splitting and conversational chains.
  - `FAISS`: For vector store and efficient retrieval.
  - `openai`: For OpenAI API interaction.
  - `dotenv`: For managing environment variables.
  -  We only tried it in `Python 12`
- **Functions:**
  - `get_data(ship_name)`: Retrieves reviews for the specified ship.
  - `get_text_chunks(raw_text)`: Splits the raw text into manageable chunks.
  - `initialize_vectorstore(chunks, force_refresh=False)`: Initializes or refreshes the vector store.
  - `get_conversation_chain(vectorstore)`: Creates a conversational chain with memory.
  - `handle_userinput(user_question)`: Handles the user's input and updates the chat history.

- **Main Function:**
  - The `main()` function sets up the Streamlit app, handles user input for ship names and questions, and manages the conversation state.

## Thanks
I highly inspired by @alehandro-ao's code. Please check out his original work;
Github: https://github.com/alejandro-ao/ask-multiple-pdfs
Youtube: https://www.youtube.com/watch?v=dXxQ0LR-3Hg&list=LL&index=4

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

