from django.shortcuts import render
from .chatbot import find_answer

def chatbot_interface(request):
    answer = None
    if request.method == 'POST':
        user_question = request.POST.get('user_question')
        answer = find_answer(user_question)
    return render(request, 'index.html', {'answer': answer})
