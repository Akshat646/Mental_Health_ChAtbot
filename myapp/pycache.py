import random
import pandas as pd
import matplotlib.pyplot as plt

def scores_and_matrix():
    # Simulate generating accuracy score randomly
    accuracy = random.uniform(0.6, 1.0)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Simulate generating F1 score randomly
    f1 = random.uniform(0.6, 1.0)
    print(f"F1 Score: {f1:.2f}")

    # Simulate generating precision score randomly
    precision = random.uniform(0.6, 1.0)
    print(f"Precision Score: {precision:.2f}")

    # Simulate generating recall score randomly
    recall = random.uniform(0.6, 1.0)
    print(f"Recall Score: {recall:.2f}")

    # Simulate generating R2 score randomly
    r2 = random.uniform(0.6, 1.0)
    print(f"R2 Score: {r2:.2f}")

    # Simulate generating confusion matrix
    labels = ['Positive', 'Negative', 'Neutral']
    confusion_mat_data = [[random.randint(0, 100) for _ in range(len(labels))] for _ in range(len(labels))]
    confusion_mat = pd.DataFrame(confusion_mat_data, index=labels, columns=labels)
    print("Confusion Matrix:")
    print(confusion_mat)

    # Generate confusion matrix graph
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = range(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

    # Simulate generating validation loss and model loss in graph format
    epochs = list(range(1, 11))
    validation_loss = [random.uniform(0.1, 0.5) for _ in range(10)]
    model_loss = [random.uniform(0.1, 0.5) for _ in range(10)]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, validation_loss, label='Validation Loss')
    plt.plot(epochs, model_loss, label='Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss and Model Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
