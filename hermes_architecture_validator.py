import ast
import os


class HermesValidator:
    """
    Analizzatore statico per il progetto HERMES.
    Verifica la separazione MVC (Model-View-Controller) e il flusso dei dati.
    """

    def __init__(self):
        self.project_dir = os.getcwd()
        self.errors = []
        self.warnings = []
        self.successes = []

    def check_file(self, filename):
        path = os.path.join(self.project_dir, filename)
        if not os.path.exists(path):
            return

        with open(path, encoding="utf-8") as f:
            try:
                source = f.read()
                tree = ast.parse(source)
            except SyntaxError as e:
                self.errors.append(f"{filename}: ERRORE SINTASSI - {e}")
                return

        # 1. Identifica Classi Logic e View
        logic_classes = {}
        view_classes = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Euristica per identificare i ruoli delle classi dai nomi
                # Aggiunto "Context" per riconoscere AppContext come logica
                if any(x in node.name for x in ["Logic", "Manager", "Generator", "Cropper", "Context"]):
                    logic_classes[node.name] = node
                elif any(x in node.name for x in ["View", "Wizard", "App"]):
                    view_classes[node.name] = node

        # 2. Controllo Purezza Logica (Niente Tkinter)
        for name, node in logic_classes.items():
            if self._uses_tkinter(node):
                self.errors.append(f"{filename} -> {name}: VIOLAZIONE MVC! La classe Logica usa Tkinter/UI.")
            else:
                self.successes.append(f"{filename} -> {name}: Logica pulita (Nessun codice UI rilevato).")

        # 3. Controllo Integrazione (View chiama Logic)
        for name, node in view_classes.items():
            logic_calls = self._find_logic_calls(node)
            if logic_calls:
                self.successes.append(f"{filename} -> {name}: Connessa correttamente. Chiama: {logic_calls}")
            else:
                # Alcune view potrebbero essere semplici wrapper, ma è sospetto
                self.warnings.append(f"{filename} -> {name}: Nessuna chiamata esplicita a 'self.logic.*' trovata.")

    def _uses_tkinter(self, class_node):
        """Controlla se una classe usa moduli UI."""
        for node in ast.walk(class_node):
            # Cerca attributi come tk.Label, ttk.Button
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id in [
                    "tk",
                    "ttk",
                    "messagebox",
                    "filedialog",
                    "simpledialog",
                ]:
                    return True
            # Cerca chiamate dirette se importate from tkinter import ...
            if isinstance(node, ast.Name) and node.id in [
                "messagebox",
                "filedialog",
                "simpledialog",
                "Label",
                "Button",
            ]:
                return True
        return False

    def _find_logic_calls(self, class_node):
        """Cerca chiamate a metodi self.logic.metodo() o self.pm.metodo()."""
        calls = []
        for node in ast.walk(class_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    # Controlla self.logic.metodo()
                    if isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "logic":
                        calls.append(f"logic.{node.func.attr}")
                    # Controlla self.pm.metodo() (ProfileManager)
                    elif isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "pm":
                        calls.append(f"pm.{node.func.attr}")
                    # Controlla self.context.metodo() (Per i Wizard e Main App)
                    elif isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "context":
                        calls.append(f"context.{node.func.attr}")
        return list(set(calls))

    def report(self):
        print("\n" + "=" * 60)
        print("   HERMES ARCHITECTURE REPORT   ")  # Removed f-prefix
        print("=" * 60)
        print(f"Directory: {self.project_dir}\n")

        if self.errors:
            print("❌ ERRORI CRITICI (Violazioni Architettura):")
            for e in self.errors:
                print(f"  - {e}")  # Split into new line
        else:
            print("✅ Nessuna violazione critica trovata.")

        if self.warnings:
            print("\n⚠️ AVVERTIMENTI (Possibili problemi di collegamento):")
            for w in self.warnings:
                print(f"  - {w}")  # Split into new line

        print("\n🔹 VERIFICHE SUPERATE (Pattern corretti):")
        for s in self.successes:
            print(f"  - {s}")  # Split into new line
        print("\n" + "=" * 60)


if __name__ == "__main__":
    validator = HermesValidator()

    # Lista dei file da analizzare
    files = [
        "hermes_context.py",
        "hermes_entity.py",
        "hermes_eye.py",
        "hermes_filters.py",
        "hermes_human.py",
        "hermes_master_toi.py",
        "hermes_region.py",
        "hermes_reviewer.py",
        "hermes_stats.py",
        "hermes_unified.py",
    ]

    print(f"Analisi di {len(files)} script in corso...")
    for f in files:
        validator.check_file(f)

    validator.report()
