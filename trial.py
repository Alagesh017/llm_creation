

import math
import random

print("=" * 60)
print("   LLM FROM SCRATCH - MANUAL MATH ONLY")
print("=" * 60)

# ─────────────────────────────────────────────
#  DATASET (100+ samples: text → sentiment 0/1)
# ─────────────────────────────────────────────
print("\n[STEP 1] Loading dataset...")

RAW_DATA = [
    ("i love this product it is amazing",         1),
    ("this is the worst thing i ever bought",      0),
    ("absolutely fantastic experience",            1),
    ("terrible quality do not buy",                0),
    ("i am so happy with my purchase",             1),
    ("complete waste of money",                    0),
    ("great value for the price",                  1),
    ("very disappointed with the result",          0),
    ("highly recommend to everyone",               1),
    ("never buying this again",                    0),
    ("outstanding customer service",               1),
    ("product broke after one day",                0),
    ("exceeded all my expectations",               1),
    ("awful smell and poor quality",               0),
    ("best purchase i have made this year",        1),
    ("total garbage not worth a penny",            0),
    ("works perfectly right out of the box",       1),
    ("instructions are confusing and misleading",  0),
    ("really good product love the design",        1),
    ("stopped working after two uses",             0),
    ("super fast delivery and great packaging",    1),
    ("item arrived damaged and unusable",          0),
    ("loving every bit of this product",           1),
    ("horrible experience with this brand",        0),
    ("five stars without any hesitation",          1),
    ("one star because zero is not an option",     0),
    ("beautiful craftsmanship and solid build",    1),
    ("cheap plastic and falls apart quickly",      0),
    ("my whole family loves this product",         1),
    ("burned my hand on first use dangerous",      0),
    ("exactly as described in the listing",        1),
    ("nothing like the photos online misleading",  0),
    ("great gift for anyone who loves cooking",    1),
    ("battery dies after thirty minutes awful",    0),
    ("feels premium and works like a charm",       1),
    ("packaging was great product was not",        0),
    ("makes my morning routine so much better",    1),
    ("had to return it immediately not working",   0),
    ("incredibly useful and well thought out",     1),
    ("flimsy and broke on first day pathetic",     0),
    ("good quality solid and reliable",            1),
    ("totally useless did nothing as promised",    0),
    ("love the color and the material quality",    1),
    ("do not trust the reviews it is bad",         0),
    ("works as advertised super satisfied",        1),
    ("leaked everywhere made a huge mess",         0),
    ("very comfortable and easy to use daily",     1),
    ("feels like a fake knockoff product",         0),
    ("brilliant design and easy assembly",         1),
    ("missing parts and no support from seller",   0),
    ("perfect size and very lightweight",          1),
    ("too heavy and bulky to carry around",        0),
    ("the performance is simply incredible",       1),
    ("does not work with any of my devices",       0),
    ("highly durable and long lasting product",    1),
    ("rusted after just one week ridiculous",      0),
    ("excellent sound quality and clear audio",    1),
    ("buzzing noise all the time very annoying",   0),
    ("smooth operation and quiet motor",           1),
    ("loud and vibrates the whole table awful",    0),
    ("really happy with this amazing buy",         1),
    ("worst customer service ever experienced",    0),
    ("looks great and performs even better",       1),
    ("photo does not match the actual product",    0),
    ("this changed my life for the better",        1),
    ("regret buying this complete rip off",        0),
    ("sturdy well built and good material",        1),
    ("broke on first use such poor quality",       0),
    ("great ergonomic design very comfortable",    1),
    ("caused pain after short period of use",      0),
    ("amazing product fast shipping loved it",     1),
    ("slow shipping product arrived broken",       0),
    ("very intuitive and simple to operate",       1),
    ("way too complicated impossible to use",      0),
    ("bright clear display stunning visuals",      1),
    ("screen flickering from day one bad",         0),
    ("solid investment great return on value",     1),
    ("not worth the high price at all",            0),
    ("just what i needed perfect timing",          1),
    ("arrived late and completely wrong item",     0),
    ("beautiful product very elegant design",      1),
    ("looks nothing like the advertisement",       0),
    ("works great in all weather conditions",      1),
    ("stopped working in the rain terrible",       0),
    ("super easy setup took two minutes",          1),
    ("spent hours trying to set up failed",        0),
    ("child safe and very durable for kids",       1),
    ("sharp edges dangerous for children awful",   0),
    ("fresh smell and pleasant texture nice",      1),
    ("strange odor and sticky texture gross",      0),
    ("very responsive customer care team",         1),
    ("no response from seller after a week",       0),
    ("does exactly what it says on the box",       1),
    ("false advertising this product is fake",     0),
    ("great improvement over previous model",      1),
    ("downgrade from older version avoid",         0),
    ("stylish modern and functional design",       1),
    ("ugly cheap looking and poorly made",         0),
    ("long lasting battery great standby",         1),
    ("battery drains within one hour bad",         0),
]

print(f"   Total samples loaded: {len(RAW_DATA)}")
print(f"   Positive samples: {sum(1 for _, y in RAW_DATA if y == 1)}")
print(f"   Negative samples: {sum(1 for _, y in RAW_DATA if y == 0)}")

print("\n[STEP 2] Building vocabulary...")

vocab = {}
word_index = 0
for text, _ in RAW_DATA:
    for word in text.split():
        if word not in vocab:
            vocab[word] = word_index
            word_index += 1

VOCAB_SIZE = len(vocab)
print(f"   Vocabulary size: {VOCAB_SIZE} unique words")
print(f"   Sample vocab entries: { {k: vocab[k] for k in list(vocab)[:5]} }")

print("\n[STEP 3] Converting text to Bag-of-Words vectors manually...")

def text_to_bow(text, vocab, vocab_size):
    """Manual Bag-of-Words: count word occurrences."""
    vector = [0.0] * vocab_size
    for word in text.split():
        if word in vocab:
            vector[vocab[word]] += 1.0
    return vector


X_data = []
Y_data = []
for text, label in RAW_DATA:
    X_data.append(text_to_bow(text, vocab, VOCAB_SIZE))
    Y_data.append(float(label))

print(f"   Feature matrix shape: {len(X_data)} x {len(X_data[0])}")
print(f"   First label: {Y_data[0]}, Last label: {Y_data[-1]}")


print("\n[STEP 4] Splitting data into train/test (80/20)...")

random.seed(42)
indices = list(range(len(X_data)))
random.shuffle(indices)

split = int(0.8 * len(indices))
train_idx = indices[:split]
test_idx  = indices[split:]

X_train = [X_data[i] for i in train_idx]
Y_train = [Y_data[i] for i in train_idx]
X_test  = [X_data[i] for i in test_idx]
Y_test  = [Y_data[i] for i in test_idx]

print(f"   Train size: {len(X_train)}")
print(f"   Test size : {len(X_test)}")



print("\n[STEP 5] Initializing neural network weights manually...")

INPUT_SIZE  = VOCAB_SIZE
HIDDEN_SIZE = 16
OUTPUT_SIZE = 1
LEARNING_RATE = 0.01
EPOCHS = 30

def xavier_init(rows, cols):
    """Xavier weight initialization formula: W ~ Uniform(-sqrt(6/(r+c)), sqrt(6/(r+c)))"""
    limit = math.sqrt(6.0 / (rows + cols))
    return [[random.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]

def zeros(size):
    return [0.0] * size

# Layer 1: Input → Hidden
W1 = xavier_init(INPUT_SIZE, HIDDEN_SIZE)   # shape: [INPUT x HIDDEN]
b1 = zeros(HIDDEN_SIZE)                     # shape: [HIDDEN]

# Layer 2: Hidden → Output
W2 = xavier_init(HIDDEN_SIZE, OUTPUT_SIZE)  # shape: [HIDDEN x 1]
b2 = zeros(OUTPUT_SIZE)                     # shape: [1]

print(f"   W1 shape: [{INPUT_SIZE} x {HIDDEN_SIZE}]")
print(f"   b1 shape: [{HIDDEN_SIZE}]")
print(f"   W2 shape: [{HIDDEN_SIZE} x {OUTPUT_SIZE}]")
print(f"   b2 shape: [{OUTPUT_SIZE}]")
print(f"   Sample W1[0][0] = {W1[0][0]:.6f}")


# ─────────────────────────────────────────────
#  ACTIVATION FUNCTIONS (manual formulas)
# ─────────────────────────────────────────────
print("\n[STEP 6] Defining activation functions using formulas...")

def relu(x):
    """ReLU: f(x) = max(0, x)"""
    return max(0.0, x)

def relu_derivative(x):
    """ReLU derivative: f'(x) = 1 if x > 0 else 0"""
    return 1.0 if x > 0 else 0.0

def sigmoid(x):
    """Sigmoid: f(x) = 1 / (1 + e^(-x))"""
    # Clip to prevent overflow
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(sx):
    """Sigmoid derivative: f'(x) = s(x) * (1 - s(x))  [given s(x)]"""
    return sx * (1.0 - sx)

print("   relu(2.0)     =", relu(2.0))
print("   relu(-3.0)    =", relu(-3.0))
print("   sigmoid(0.0)  =", sigmoid(0.0))
print("   sigmoid(2.0)  =", round(sigmoid(2.0), 5))


# ─────────────────────────────────────────────
#  FORWARD PASS (manual matrix multiply)
# ─────────────────────────────────────────────
print("\n[STEP 7] Defining forward pass with manual dot products...")

def dot_product(vec, matrix_col):
    """Manual dot product: sum(v[i] * m[i]) for all i"""
    result = 0.0
    for i in range(len(vec)):
        result += vec[i] * matrix_col[i]
    return result

def forward(x, W1, b1, W2, b2):
    """
    Forward pass:
      z1[j] = sum_i(x[i] * W1[i][j]) + b1[j]   <- linear layer 1
      a1[j] = ReLU(z1[j])                        <- activation
      z2    = sum_j(a1[j] * W2[j][0]) + b2[0]   <- linear layer 2
      y_hat = sigmoid(z2)                         <- output probability
    """
    # Layer 1: z1 = X @ W1 + b1
    z1 = []
    for j in range(HIDDEN_SIZE):
        val = b1[j]
        for i in range(INPUT_SIZE):
            val += x[i] * W1[i][j]
        z1.append(val)

    # Activation: a1 = ReLU(z1)
    a1 = [relu(z) for z in z1]

    # Layer 2: z2 = a1 @ W2 + b2
    z2 = b2[0]
    for j in range(HIDDEN_SIZE):
        z2 += a1[j] * W2[j][0]

    # Output: y_hat = sigmoid(z2)
    y_hat = sigmoid(z2)

    return z1, a1, z2, y_hat

print("   Forward pass function defined.")
print("   Testing forward pass on sample input...")
sample_z1, sample_a1, sample_z2, sample_yhat = forward(X_train[0], W1, b1, W2, b2)
print(f"   z2 = {sample_z2:.6f},  y_hat = {sample_yhat:.6f},  true_y = {Y_train[0]}")


# ─────────────────────────────────────────────
#  LOSS FUNCTION: Binary Cross Entropy (manual)
# ─────────────────────────────────────────────
print("\n[STEP 8] Defining Binary Cross Entropy loss manually...")

def bce_loss(y_true, y_pred):
    """
    BCE formula:
      L = -[ y * log(y_hat) + (1-y) * log(1-y_hat) ]
    """
    eps = 1e-9   # avoid log(0)
    y_pred = max(eps, min(1.0 - eps, y_pred))
    return -(y_true * math.log(y_pred) + (1.0 - y_true) * math.log(1.0 - y_pred))

print(f"   BCE(y=1, y_hat=0.9) = {bce_loss(1, 0.9):.5f}")
print(f"   BCE(y=1, y_hat=0.1) = {bce_loss(1, 0.1):.5f}")
print(f"   BCE(y=0, y_hat=0.1) = {bce_loss(0, 0.1):.5f}")


# ─────────────────────────────────────────────
#  BACKPROPAGATION (manual chain rule)
# ─────────────────────────────────────────────
print("\n[STEP 9] Defining backpropagation using chain rule formulas...")

def backward_and_update(x, y_true, z1, a1, y_hat, W1, b1, W2, b2, lr):
    """
    Backprop chain rule:
      dL/dy_hat  = -(y/y_hat - (1-y)/(1-y_hat))
      dy_hat/dz2 = sigmoid'(z2) = y_hat*(1-y_hat)
      delta2     = dL/dz2 = dL/dy_hat * dy_hat/dz2

      dL/dW2[j]  = delta2 * a1[j]
      dL/db2     = delta2

      dL/da1[j]  = delta2 * W2[j][0]
      dL/dz1[j]  = dL/da1[j] * relu'(z1[j])

      dL/dW1[i][j] = dL/dz1[j] * x[i]
      dL/db1[j]    = dL/dz1[j]
    """
    eps = 1e-9
    yh = max(eps, min(1.0 - eps, y_hat))

    # Output layer gradient
    dL_dyhat = -(y_true / yh - (1.0 - y_true) / (1.0 - yh))
    dyhat_dz2 = yh * (1.0 - yh)          # sigmoid derivative
    delta2 = dL_dyhat * dyhat_dz2

    # Gradients for W2 and b2
    dW2 = [delta2 * a1[j] for j in range(HIDDEN_SIZE)]
    db2 = delta2

    # Hidden layer gradient
    delta1 = []
    for j in range(HIDDEN_SIZE):
        dL_da1j = delta2 * W2[j][0]
        dL_dz1j = dL_da1j * relu_derivative(z1[j])
        delta1.append(dL_dz1j)

    # Update W2 and b2
    for j in range(HIDDEN_SIZE):
        W2[j][0] -= lr * dW2[j]
    b2[0] -= lr * db2

    # Update W1 and b1 (only update non-zero input features for speed)
    for i in range(INPUT_SIZE):
        if x[i] != 0.0:
            for j in range(HIDDEN_SIZE):
                W1[i][j] -= lr * delta1[j] * x[i]
    for j in range(HIDDEN_SIZE):
        b1[j] -= lr * delta1[j]

    return W1, b1, W2, b2

print("   Backpropagation function defined.")


# ─────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────
print("\n[STEP 10] Starting training loop...")
print(f"   Epochs: {EPOCHS}  |  Learning rate: {LEARNING_RATE}  |  Train samples: {len(X_train)}")
print("-" * 60)

epoch_losses = []

for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    correct = 0

    # Shuffle training data each epoch
    train_pairs = list(zip(X_train, Y_train))
    random.shuffle(train_pairs)

    for x, y_true in train_pairs:
        # Forward pass
        z1, a1, z2, y_hat = forward(x, W1, b1, W2, b2)

        # Compute loss
        loss = bce_loss(y_true, y_hat)
        total_loss += loss

        # Prediction (threshold at 0.5)
        pred = 1 if y_hat >= 0.5 else 0
        if pred == int(y_true):
            correct += 1

        # Backward pass and weight update
        W1, b1, W2, b2 = backward_and_update(x, y_true, z1, a1, y_hat, W1, b1, W2, b2, LEARNING_RATE)

    avg_loss = total_loss / len(X_train)
    accuracy = correct / len(X_train) * 100.0
    epoch_losses.append(avg_loss)

    if epoch % 5 == 0 or epoch == 1:
        print(f"   Epoch [{epoch:2d}/{EPOCHS}]  Loss: {avg_loss:.5f}  Train Accuracy: {accuracy:.2f}%")

print("-" * 60)
print("   Training complete!")


# ─────────────────────────────────────────────
#  EVALUATION ON TEST SET
# ─────────────────────────────────────────────
print("\n[STEP 11] Evaluating on test set...")

test_correct = 0
test_total_loss = 0.0
true_pos = 0
false_pos = 0
true_neg = 0
false_neg = 0

for x, y_true in zip(X_test, Y_test):
    _, _, _, y_hat = forward(x, W1, b1, W2, b2)
    loss = bce_loss(y_true, y_hat)
    test_total_loss += loss

    pred = 1 if y_hat >= 0.5 else 0
    actual = int(y_true)

    if pred == actual:
        test_correct += 1
    if pred == 1 and actual == 1:
        true_pos += 1
    elif pred == 1 and actual == 0:
        false_pos += 1
    elif pred == 0 and actual == 0:
        true_neg += 1
    elif pred == 0 and actual == 1:
        false_neg += 1

test_accuracy = test_correct / len(X_test) * 100.0
test_avg_loss = test_total_loss / len(X_test)

print(f"   Test Samples  : {len(X_test)}")
print(f"   Test Loss     : {test_avg_loss:.5f}")
print(f"   Test Accuracy : {test_accuracy:.2f}%")
print(f"   True Positives : {true_pos}")
print(f"   True Negatives : {true_neg}")
print(f"   False Positives: {false_pos}")
print(f"   False Negatives: {false_neg}")


# ─────────────────────────────────────────────
#  METRICS: Precision, Recall, F1 (manual formulas)
# ─────────────────────────────────────────────
print("\n[STEP 12] Computing Precision, Recall, F1 manually...")

def precision(tp, fp):
    """Precision = TP / (TP + FP)"""
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall(tp, fn):
    """Recall = TP / (TP + FN)"""
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_score(prec, rec):
    """F1 = 2 * (Precision * Recall) / (Precision + Recall)"""
    return 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

prec  = precision(true_pos, false_pos)
rec   = recall(true_pos, false_neg)
f1    = f1_score(prec, rec)

print(f"   Precision : {prec:.4f}")
print(f"   Recall    : {rec:.4f}")
print(f"   F1 Score  : {f1:.4f}")


# ─────────────────────────────────────────────
#  LOSS CURVE SUMMARY (manual)
# ─────────────────────────────────────────────
print("\n[STEP 13] Loss curve summary across epochs...")

print("   Epoch | Avg Loss")
print("   ------|----------")
for i, l in enumerate(epoch_losses, 1):
    bar = "#" * int(l * 20)
    print(f"   {i:5d} | {l:.5f}  {bar}")

loss_drop = epoch_losses[0] - epoch_losses[-1]
print(f"\n   Initial loss : {epoch_losses[0]:.5f}")
print(f"   Final loss   : {epoch_losses[-1]:.5f}")
print(f"   Loss reduced by: {loss_drop:.5f}")


# ─────────────────────────────────────────────
#  LIVE INFERENCE
# ─────────────────────────────────────────────
print("\n[STEP 14] Running live inference on new sentences...")

test_sentences = [
    "i love this amazing product highly recommend",
    "terrible quality waste of money do not buy",
    "works great very happy with the purchase",
    "broke after one use complete garbage",
    "fantastic build quality excellent performance",
]

print("   Input Sentence → Prediction (confidence)")
print("   " + "-" * 55)
for sentence in test_sentences:
    vec = text_to_bow(sentence, vocab, VOCAB_SIZE)
    _, _, _, y_hat = forward(vec, W1, b1, W2, b2)
    label = "POSITIVE 😊" if y_hat >= 0.5 else "NEGATIVE 😞"
    print(f"   '{sentence[:40]}...'")
    print(f"    → {label}  (confidence: {y_hat:.4f})")
    print()


# ─────────────────────────────────────────────
#  WEIGHT STATISTICS (manual mean, std)
# ─────────────────────────────────────────────
print("[STEP 15] Final weight statistics (manual mean & std)...")

all_w2 = [W2[j][0] for j in range(HIDDEN_SIZE)]

w2_mean = sum(all_w2) / len(all_w2)
w2_variance = sum((w - w2_mean) ** 2 for w in all_w2) / len(all_w2)
w2_std = math.sqrt(w2_variance)

print(f"   W2 mean : {w2_mean:.6f}")
print(f"   W2 std  : {w2_std:.6f}")
print(f"   W2 min  : {min(all_w2):.6f}")
print(f"   W2 max  : {max(all_w2):.6f}")

print("\n" + "=" * 60)
print("   TRAINING DONE! LLM built fully from scratch.")
print("   No sklearn / torch / numpy used. Pure Python math!")
print("=" * 60)