import joblib
import subprocess

MODEL_FILE = "regression.joblib"
OUTPUT_C_FILE = "prediction_model.c"
OUTPUT_EXECUTABLE = "prediction_model"

try:
    model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    print(f"Erreur : Le fichier modèle '{MODEL_FILE}' n'a pas été trouvé.")
    print("Veuillez d'abord lancer le script 'train_model.py'.")
    exit()

model_type = type(model).__name__

# ############# REGRESSION LINEAIRE #############

if "LinearRegression" in model_type:
    intercept = model.intercept_
    coefficients = model.coef_
    n_features = len(coefficients)
    coeffs_str = ", ".join([str(c) + "f" for c in coefficients])
    sample_features_c = "{100.0f, 3.0f, 1.0f}"

    c_code_template = f"""
#include <stdio.h>

const float intercept = {intercept}f;
const float coefficients[{n_features}] = {{ {coeffs_str} }};

/**
 * @brief Calcule la prédiction d'un modèle de régression linéaire.
 * @param features Tableau de flottants contenant les caractéristiques.
 * @param n_feature Le nombre de caractéristiques.
 * @return La prédiction calculée.
 */
float prediction(float* features, int n_feature) {{
    if (n_feature != {n_features}) {{
        printf("Erreur: Le nombre de features (%d) ne correspond pas au modèle ({n_features})\\n", n_feature);
        return 0.0f;
    }}
    
    float pred = intercept;
    for (int i = 0; i < n_feature; i++) {{
        pred += coefficients[i] * features[i];
    }}
    return pred;
}}

int main() {{
    float sample_features[{n_features}] = {sample_features_c};
    float result = prediction(sample_features, {n_features});
    printf("Prédiction du modèle C pour l'exemple : %f\\n", result);
    return 0;
}}
"""

# ############# REGRESSION LOGISTIQUE #############

elif "LogisticRegression" in model_type:
    # Code pour la régression logistique
    n_classes = len(model.classes_)

    if n_classes == 2:
        # Classification binaire - construire le tableau thetas avec intercept en position 0
        intercept = model.intercept_[0]
        coefficients = model.coef_[0]
        n_features = len(coefficients)

        # Construire le tableau thetas: [intercept, coef1, coef2, ..., coefN]
        thetas = [intercept] + list(coefficients)
        thetas_str = ", ".join([str(t) + "f" for t in thetas])
        n_parameters = len(thetas)

        sample_features_c = "{1.0f, 2.0f, 3.0f}"

        c_code_template = f"""
#include <stdio.h>
#include <math.h>

const int n_features = {n_features};
const float thetas[{n_parameters}] = {{ {thetas_str} }};

float exp_approx(float x, int n_term) {{
    float sum = 1.0f;
    float term = 1.0f;
    
    for (int i = 1; i <= n_term; i++) {{
        term = term * x / i;
        sum = sum + term;
    }}
    
    return sum;
}}

/**
 * @brief Fonction sigmoid.
 * @param x La valeur d'entrée.
 * @return La valeur sigmoid entre 0 et 1.
 */
float sigmoid(float x) {{
    return 1.0 / (1.0 + exp_approx(-x, 10));
}}

/**
 * @brief Calcule la probabilité de la classe positive selon la régression logistique.
 * @param features Tableau de flottants contenant les caractéristiques.
 * @param thetas Tableau des paramètres (theta[0] = intercept, theta[1..n] = coefficients).
 * @param n_parameter Le nombre de paramètres (features).
 * @return La probabilité entre 0 et 1.
 */
float logistic_regression(float* features, float* thetas, int n_parameter) {{
    float z = thetas[0];
    
    for (int i = 0; i < n_parameter; i++) {{
        z += features[i] * thetas[i + 1];
    }}

    return sigmoid(z);
}}

/**
 * @brief Prédit la classe (0 ou 1).
 * @param features Tableau de flottants contenant les caractéristiques.
 * @param n_features Le nombre de caractéristiques.
 * @return La classe prédite (0 ou 1).
 */
int prediction(float* features, int n_features) {{
    if (n_features != {n_features}) {{
        printf("Erreur: Le nombre de features (%d) ne correspond pas au modèle ({n_features})\\n", n_features);
        return 0;
    }}
    
    float proba = logistic_regression(features, (float*)thetas, n_features);
    return (proba >= 0.5f) ? 1 : 0;
}}

int main() {{
    float sample_features[n_features] = {sample_features_c};
    int result = prediction(sample_features, n_features);
    float proba = logistic_regression(sample_features, (float*)thetas, n_features);
    printf("Prédiction du modèle C pour l'exemple : classe %d (probabilité %.4f)\\n", result, proba);
    return 0;
}}
"""
    else:
        intercepts = model.intercept_
        coefficients = model.coef_
        n_features = coefficients.shape[1]
        classes = model.classes_

        # Construire les tableaux thetas pour chaque classe
        thetas_arrays = []
        for i in range(n_classes):
            thetas_class = [intercepts[i]] + list(coefficients[i])
            thetas_arrays.append(thetas_class)

        # Génération des tableaux pour chaque classe
        thetas_str = ""
        for i, thetas in enumerate(thetas_arrays):
            thetas_str += "    {" + ", ".join([str(t) + "f" for t in thetas]) + "}"
            if i < len(thetas_arrays) - 1:
                thetas_str += ",\n"

        classes_str = ", ".join([str(c) for c in classes])
        sample_features_c = "{1.0f, 2.0f, 3.0f, 4.0f}"

        c_code_template = f"""
#include <stdio.h>
#include <math.h>

const int n_classes = {n_classes};
const int n_features = {n_features};
const float thetas[{n_classes}][{n_features + 1}] = {{
{thetas_str}
}};
const int classes[{n_classes}] = {{ {classes_str} }};

float exp_approx(float x, int n_term) {{
    float sum = 1.0f;
    float term = 1.0f;
    
    for (int i = 1; i <= n_term; i++) {{
        term = term * x / i;
        sum = sum + term;
    }}
    
    return sum;
}}

/**
 * @brief Fonction sigmoid.
 * @param x La valeur d'entrée.
 * @return La valeur sigmoid entre 0 et 1.
 */
float sigmoid(float x) {{
    return 1.0 / (1.0 + exp_approx(-x, 10));
}}

/**
 * @brief Calcule la probabilité selon la régression logistique.
 * @param features Tableau de flottants contenant les caractéristiques.
 * @param thetas Tableau des paramètres (theta[0] = intercept, theta[1..n] = coefficients).
 * @param n_parameter Le nombre de paramètres (features).
 * @return La probabilité entre 0 et 1.
 */
float logistic_regression(float* features, float* thetas, int n_parameter) {{
    float z = thetas[0];
    
    for (int i = 0; i < n_parameter; i++) {{
        z += features[i] * thetas[i + 1];
    }}

    return sigmoid(z);
}}

/**
 * @brief Calcule le softmax pour la classification multiclasse.
 * @param logits Tableau des valeurs z pour chaque classe.
 * @param probas Tableau de sortie pour les probabilités.
 * @param n_classes Nombre de classes.
 */
void softmax(float* logits, float* probas, int n_classes) {{
    float max_logit = logits[0];
    for (int i = 1; i < n_classes; i++) {{
        if (logits[i] > max_logit) max_logit = logits[i];
    }}
    
    float sum = 0.0f;
    for (int i = 0; i < n_classes; i++) {{
        probas[i] = exp_approx(logits[i] - max_logit);
        sum += probas[i];
    }}
    
    for (int i = 0; i < n_classes; i++) {{
        probas[i] /= sum;
    }}
}}

/**
 * @brief Calcule les probabilités pour toutes les classes.
 * @param features Tableau de flottants contenant les caractéristiques.
 * @param probas Tableau de sortie pour les probabilités de chaque classe.
 */
void predict_proba(float* features, float* probas) {{
    float logits[n_classes];
    
    for (int c = 0; c < n_classes; c++) {{
        logits[c] = logistic_regression(features, (float*)thetas[c], n_features);
    }}
    
    softmax(logits, probas, n_classes);
}}

/**
 * @brief Prédit la classe.
 * @param features Tableau de flottants contenant les caractéristiques.
 * @return La classe prédite.
 */
int prediction(float* features) {{
    float probas[n_classes];
    predict_proba(features, probas);
    
    int max_idx = 0;
    float max_proba = probas[0];
    for (int i = 1; i < n_classes; i++) {{
        if (probas[i] > max_proba) {{
            max_proba = probas[i];
            max_idx = i;
        }}
    }}
    
    return classes[max_idx];
}}

int main() {{
    float sample_features[n_features] = {sample_features_c};
    int result = prediction(sample_features);
    
    float probas[n_classes];
    predict_proba(sample_features, probas);
    
    printf("Prédiction du modèle C pour l'exemple : classe %d\\n", result);
    printf("Probabilités : ");
    for (int i = 0; i < n_classes; i++) {{
        printf("classe %d: %.4f  ", classes[i], probas[i]);
    }}
    printf("\\n");
    return 0;
}}
"""

# ############# ARBRE DE DÉCISION #############

elif "DecisionTree" in model_type:
    tree = model.tree_
    n_features = tree.n_features

    def generate_tree_code(node_id, depth=0):
        """Génère récursivement le code C pour l'arbre de décision."""
        indent = "    " * depth

        # Vérifier si c'est une feuille
        if tree.children_left[node_id] == tree.children_right[node_id]:
            value = tree.value[node_id]
            if hasattr(model, "classes_"):
                # Classification
                predicted_class = model.classes_[value.argmax()]
                return f"{indent}return {predicted_class};\n"
            else:
                # Régression
                predicted_value = value[0, 0]
                return f"{indent}return {predicted_value}f;\n"

        # Noeud de décision
        feature_id = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        code = f"{indent}if (features[{feature_id}] <= {threshold}f) {{\n"
        code += generate_tree_code(left_child, depth + 1)
        code += f"{indent}}} else {{\n"
        code += generate_tree_code(right_child, depth + 1)
        code += f"{indent}}}\n"

        return code

    tree_function_body = generate_tree_code(0, 1)

    # Déterminer le type de retour
    if hasattr(model, "classes_"):
        return_type = "int"
        sample_features = "{1.0f, 2.0f, 3.0f, 4.0f}"
    else:
        return_type = "float"
        sample_features = "{1.0f, 2.0f, 3.0f, 4.0f}"

    c_code_template = f"""
#include <stdio.h>

/**
 * @brief Prédit la classe/valeur selon l'arbre de décision.
 * @param features Tableau de flottants contenant les caractéristiques.
 * @param n_features Le nombre de caractéristiques.
 * @return La prédiction (classe ou valeur).
 */
{return_type} prediction(float* features, int n_features) {{
    if (n_features != {n_features}) {{
        printf("Erreur: Le nombre de features (%d) ne correspond pas au modèle ({n_features})\\n", n_features);
        return 0;
    }}
    
{tree_function_body}}}

int main() {{
    float sample_features[{n_features}] = {sample_features};
    {return_type} result = prediction(sample_features, {n_features});
    printf("Prédiction du modèle C pour l'exemple : %{"d" if return_type == "int" else "f"}\\n", result);
    return 0;
}}
"""

else:
    print(f"Erreur : Type de modèle non supporté : {model_type}")
    print(
        "Seuls LinearRegression et DecisionTreeClassifier/DecisionTreeRegressor sont supportés."
    )
    exit()

with open(OUTPUT_C_FILE, "w") as f:
    f.write(c_code_template)

print(f">>> Code C généré et sauvegardé dans '{OUTPUT_C_FILE}'")
print("-" * 30)

# Compilation
compile_command = f"gcc {OUTPUT_C_FILE} -o {OUTPUT_EXECUTABLE}"
print(f"Pour compiler, lancez la commande suivante :\n\n  {compile_command}\n")

try:
    print("Tentative de compilation automatique...")
    subprocess.run(compile_command.split(), check=True)
    print(f">>> Compilation réussie ! Exécutable créé : '{OUTPUT_EXECUTABLE}'")
    print(f"Pour l'exécuter : ./{OUTPUT_EXECUTABLE}")
except (subprocess.CalledProcessError, FileNotFoundError):
    print(
        "\n/!\\ La compilation automatique a échoué. Assurez-vous que 'gcc' est installé et dans votre PATH."
    )
