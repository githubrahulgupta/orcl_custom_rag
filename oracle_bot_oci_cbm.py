#
# Streamlit App to demo OCI AI GenAI
# this is the main code, with the UI
#
import streamlit as st

# this function initialise the rag chain, creating retriever, llm and chain
# from init_rag_streamlit_oci_cbm import initialize_rag_chain, get_answer
from rag_chain import initialize_rag_chain, get_answer, clear_conv_memory

#
# Configs
#


def reset_conversation():
    # Delete all the items in Session state
    # for key in st.session_state.keys():
    #     del st.session_state[key]
    st.session_state.messages = []
    clear_conv_memory()


#
# Main
#
st.title("OCI Generative AI Bot powered by RAG")

# Added reset button
st.button("Clear Chat History", on_click=reset_conversation)

# Initialize chat history
if "messages" not in st.session_state:
    reset_conversation()

# init RAG
rag_chain = initialize_rag_chain()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # st.markdown(type(message["content"]))
        st.markdown(message["content"])

# React to user input
if question := st.chat_input("Hello, how can I help you?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(question)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})

    # here we call OCI genai...

    try:
        print("...")
        response = get_answer(rag_chain, question)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response["answer"])
            # st.markdown("***References***")
            # st.markdown(":green[***References***]")
            # st.subheader('_References_', divider='rainbow')
            # expander = st.expander("References")

            # for item in response["source_documents"]:
            #     # st.markdown(f'{item}\n')
            #     expander.write(f'{item}\n')

            # for i, item in enumerate(response["source_documents"]):
            #     expander.write(f"\n{'-' * 100}\n")
            #     expander.write(f"Reference {i+1}: ")
            #     expander.write(item.page_content)
            #     expander.write(f"Metadata {i+1}: ")
            #     expander.write(item.metadata)

            with st.expander("References"):
                for i, item in enumerate(response["source_documents"]):
                    st.markdown(f"***Reference {i+1}:***")
                    st.write(item.page_content)
                    st.markdown(f"***Metadata {i+1}:***")
                    st.write(item.metadata)
                    # st.write(f"\n{'-' * 100}\n")
                    st.divider()
                
            # st.markdown(response["source_documents"])

        # Add assistant response to chat history
        # st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
        # st.session_state.messages.append({"role": "assistant", "content": response["source_documents"]})

    except Exception as e:
        st.error("An error occurred: " + str(e))
