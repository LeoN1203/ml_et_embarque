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
