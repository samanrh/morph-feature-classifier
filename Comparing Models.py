import math
import random
from collections import defaultdict

# ----- Perceptron Model (Lab 4.1) -----
class Perceptron_Mark1():
    def __init__(self, lr=1, epochs=3):
        self.lr = float(lr)
        self.epochs = epochs
        self.weights = defaultdict(float)
        self.labels = set()
        self.train_data_path = r"C:\Users\saman\Desktop\mini-paper\swe-train"
        self.test_data_path = r"C:\Users\saman\Desktop\mini-paper\swe-test"

        with open(self.train_data_path, "r", encoding="utf-8") as f:
            self.train_data = [line.strip() for line in f if line.strip()]
        
    def extract_features(self, word, pos):
        features = set()
        for i in range(1, 6):
            if len(word) >= i:
                prefix = word[:i]
                suffix = word[-i:]
                features.add(f"{pos} prefix={prefix}")
                features.add(f"{pos} suffix={suffix}")
        return features

    def predict(self, word, possible_tags, pos):
        scores = defaultdict(float)
        features = self.extract_features(word, pos)
        for tagg in sorted(list(possible_tags)):
            for feat in features:
                scores[tagg] += self.weights.get((feat, tagg), 0.0)
        return max(scores, key=scores.get)

    def train(self):
        for epoch in range(self.epochs):
            print(f"Perceptron Epoch {epoch + 1} ...")
            for line in self.train_data:
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                _, word, gold_tag = parts
                pos = gold_tag.split(";")[0]
                gold_tag = ";".join(gold_tag.split(";")[1:])
                word = word.replace(" ", "_")

                self.labels.add(gold_tag)
                predicted_tag = self.predict(word, self.labels, pos)

                if predicted_tag != gold_tag:
                    features = self.extract_features(word, pos)
                    for feat in features:
                        self.weights[(feat, gold_tag)] += self.lr
                        self.weights[(feat, predicted_tag)] -= self.lr

    def test_with_predictions(self):
        predictions = []
        golds = []
        with open(self.test_data_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                _, word, gold_tag = parts
                pos = gold_tag.split(";")[0]
                gold_tag = ";".join(gold_tag.split(";")[1:])
                word = word.replace(" ", "_")

                pred_tag = self.predict(word, self.labels, pos)
                predictions.append(pred_tag)
                golds.append(gold_tag)

        correct = sum([p == g for p, g in zip(predictions, golds)])
        accuracy = correct / len(golds) if golds else 0
        print(f"Perceptron Accuracy: {accuracy*100:.2f}%")
        return predictions, golds

# ----- MLR Model -----
class MLR_Mark1():
    def __init__(self, lr=1.0, epochs=3):
        self.lr = float(lr)
        self.epochs = epochs
        self.weights = defaultdict(float)
        self.labels = set()
        self.train_data_path = r"C:\Users\saman\Desktop\mini-paper\swe-train"
        self.test_data_path = r"C:\Users\saman\Desktop\mini-paper\swe-test"

        with open(self.train_data_path, "r", encoding="utf-8") as f:
            self.train_data = [line.strip() for line in f if line.strip()]

    def extract_features(self, word, pos):
        features = set()
        for i in range(1, 6):
            if len(word) >= i:
                prefix = word[:i]
                suffix = word[-i:]
                features.add(f"{pos} prefix={prefix}")
                features.add(f"{pos} suffix={suffix}")
        return features

    def softmax(self, scores):
        max_score = max(scores.values())
        exp_scores = {y: math.exp(scores[y] - max_score) for y in scores}
        total = sum(exp_scores.values())
        return {y: exp_scores[y] / total for y in scores}

    def predict(self, word, possible_tags, pos):
        features = self.extract_features(word, pos)
        scores = {}
        for tagg in possible_tags:
            scores[tagg] = sum(self.weights.get((feat, tagg), 0.0) for feat in features)
        probs = self.softmax(scores)
        return max(probs, key=probs.get), probs

    def train(self):
        for epoch in range(self.epochs):
            print(f"MLR Epoch {epoch+1} ...")
            for line in self.train_data:
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                _, word, gold_tag = parts
                pos = gold_tag.split(";")[0]
                gold_tag = ";".join(gold_tag.split(";")[1:])
                word = word.replace(" ", "_")

                self.labels.add(gold_tag)
                predicted_tag, probs = self.predict(word, self.labels, pos)
                features = self.extract_features(word, pos)

                for label in self.labels:
                    error = (1.0 if label == gold_tag else 0.0) - probs.get(label, 0.0)
                    for feat in features:
                        self.weights[(feat, label)] += self.lr * error

    def test_with_predictions(self):
        predictions = []
        golds = []
        with open(self.test_data_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                _, word, gold_tag = parts
                pos = gold_tag.split(";")[0]
                gold_tag = ";".join(gold_tag.split(";")[1:])
                word = word.replace(" ", "_")

                pred_tag, _ = self.predict(word, self.labels, pos)
                predictions.append(pred_tag)
                golds.append(gold_tag)

        correct = sum([p == g for p, g in zip(predictions, golds)])
        accuracy = correct / len(golds) if golds else 0
        print(f"MLR Accuracy: {accuracy*100:.2f}%")
        return predictions, golds

# ----- Paired Bootstrap Test -----
def paired_bootstrap(preds1, preds2, golds, iterations=5000):
    diffs = []
    n = len(golds)
    for _ in range(iterations):
        indices = [random.randint(0, n-1) for _ in range(n)]
        acc1 = sum([preds1[i] == golds[i] for i in indices]) / n
        acc2 = sum([preds2[i] == golds[i] for i in indices]) / n
        diffs.append(acc1 - acc2)
    diffs.sort()
    lower = diffs[int(0.025 * iterations)]
    upper = diffs[int(0.975 * iterations)]
    avg_diff = sum(diffs) / iterations
    print(f"\nBootstrap Mean Difference (Model1 - Model2): {avg_diff:.4f}")
    print(f"95% Confidence Interval: [{lower:.4f}, {upper:.4f}]")

# ----- Run -----
if __name__ == '__main__':
    print("\n--- Perceptron ---")
    perceptron_model = Perceptron_Mark1(lr=1, epochs=3)
    perceptron_model.train()
    p_preds, golds = perceptron_model.test_with_predictions()

    print("\n--- MLR ---")
    mlr_model = MLR_Mark1(lr=1, epochs=3)
    mlr_model.train()
    m_preds, _ = mlr_model.test_with_predictions()

    # Paired Bootstrap Test
    print("\n--- Paired Bootstrap Test (Perceptron - MLR) ---")
    paired_bootstrap(p_preds, m_preds, golds)
