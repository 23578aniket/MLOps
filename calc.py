import streamlit as st
st.header("Basic Calculator")

def square(num):
    return num*num

num = st.number_input("Insert a Number")
st.write("The current number is ", num)

#how to call a function - on demand function

if st.button("Calculate Square"):
    result = square(num)

    st.text(result)

