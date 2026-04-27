from chatboot.data import training_data 
from chatboot.model import buid_and_train_model, load_model, predict_answer
def main():
    model , vectorizer, unique_answers = load_model()
    if model is None:
        model, vectorizer, unique_answers = buid_and_train_model(training_data)
print("\n 🤖 chatboot listo. escribe'salir' para terminar." )
while True:   
    user_input = input("Tú: ").strip()
    if user_input.lower() in {"salir","exit","quit"}:
        print("boot ¡Hasta pronto!")
        break   
    response = predict_answer(model, vectorizer, unique_answers, user)
    print(f"boot: {response}")
if __name__ == "__main__":   
    main()
    