from flask import Flask, request, jsonify
from textwrap import wrap
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

app = Flask(__name__)

the_model = 'mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es'
tokenizer = AutoTokenizer.from_pretrained(the_model, do_lower_case=False)
model = AutoModelForQuestionAnswering.from_pretrained(the_model)
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

@app.route('/procesar_comentario', methods=['POST'])
def procesar_comentario():
    comentario = request.json['comentario']
    pregunta = "¿De qué tema se habla?"
    salida = nlp({'question': pregunta, 'context': comentario})
    return jsonify({"respuesta": salida['answer']})

if __name__ == '__main__':
    app.run(debug=True)
