import os
import shutil
import sys
import tempfile
import unittest

# Aggiungi la directory corrente al path per permettere gli import
sys.path.append(os.getcwd())


class TestHermesLogic(unittest.TestCase):
    """
    Testa che le classi Logic di TUTTI i moduli possano essere istanziate
    e usate senza avviare l'interfaccia grafica (Tkinter).
    Verifica la "Headless Compliance" dell'intera suite.
    """

    test_dir: str

    @classmethod
    def setUpClass(cls):
        # Crea una directory temporanea per i test che richiedono path di file
        cls.test_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        # Rimuovi la directory temporanea alla fine
        shutil.rmtree(cls.test_dir)

    def test_00_context_init(self):
        """Verifica hermes_context.py (AppContext)"""
        try:
            from hermes_context import AppContext

            # AppContext è il cuore del sistema, deve caricarsi senza GUI
            ctx = AppContext()
            self.assertIsNotNone(ctx)
            # Verifica che abbia caricato le impostazioni di base
            self.assertTrue(hasattr(ctx, "participants"))
            print("✅ hermes_context: AppContext inizializzato.")
        except Exception as e:
            self.fail(f"Errore hermes_context: {e}")

    def test_01_human_logic(self):
        """Verifica hermes_human.py (PoseEstimatorLogic)"""
        try:
            from hermes_human import PoseEstimatorLogic

            logic = PoseEstimatorLogic()
            self.assertIsNotNone(logic)
            # Verifica che il metodo di download sia accessibile
            self.assertTrue(hasattr(logic, "download_model"))
            print("✅ hermes_human: PoseEstimatorLogic inizializzato.")
        except Exception as e:
            self.fail(f"Errore hermes_human: {e}")

    def test_02_master_toi_logic(self):
        """Verifica hermes_master_toi.py (MasterToiLogic & ProfileManager)"""
        try:
            from hermes_master_toi import MasterToiLogic, ProfileManager, ProfileWizardLogic

            # Test ProfileManager con cartella temporanea
            pm = ProfileManager(os.path.join(self.test_dir, "profiles"))
            self.assertIsNotNone(pm)

            # Test Logic Principale
            logic = MasterToiLogic(pm)
            self.assertIsNotNone(logic)

            # Test Wizard Logic
            wiz = ProfileWizardLogic(os.path.join(self.test_dir, "profiles"))
            self.assertIsNotNone(wiz)

            print("✅ hermes_master_toi: Logica inizializzata.")
        except Exception as e:
            self.fail(f"Errore hermes_master_toi: {e}")

    def test_03_entity_logic(self):
        """Verifica hermes_entity.py (IdentityLogic & HistoryManager)"""
        try:
            from hermes_entity import HistoryManager, IdentityLogic

            logic = IdentityLogic(fps=30.0)
            hist = HistoryManager()

            self.assertEqual(logic.fps, 30.0)
            self.assertIsNotNone(hist)

            # Test rapido di una funzione logica pura
            logic.tracks = {
                1: {"frames": [0, 1], "boxes": [[0, 0, 10, 10], [1, 1, 11, 11]], "role": "Ignore", "merged_from": [1]}
            }
            snap = logic.get_data_snapshot()
            self.assertIn(1, snap[0])

            print("✅ hermes_entity: Logica inizializzata e testata.")
        except Exception as e:
            self.fail(f"Errore hermes_entity: {e}")

    def test_04_region_logic(self):
        """Verifica hermes_region.py (RegionLogic & AOIProfileManager)"""
        try:
            from hermes_region import AOIProfileManager, RegionLogic

            logic = RegionLogic()
            pm = AOIProfileManager(folder=os.path.join(self.test_dir, "aoi_profiles"))

            self.assertIsNotNone(logic)
            self.assertIsNotNone(pm)

            # Verifica che la logica abbia le strutture dati pronte
            self.assertEqual(logic.pose_data, {})
            self.assertEqual(logic.manual_overrides, {})

            print("✅ hermes_region: Logica inizializzata.")
        except Exception as e:
            self.fail(f"Errore hermes_region: {e}")

    def test_05_eye_logic(self):
        """Verifica hermes_eye.py (GazeLogic)"""
        try:
            from hermes_eye import GazeLogic

            logic = GazeLogic()
            self.assertIsNotNone(logic)
            # Verifica esistenza metodo statico di calcolo
            self.assertTrue(hasattr(logic, "calculate_hit"))
            print("✅ hermes_eye: GazeLogic inizializzato.")
        except Exception as e:
            self.fail(f"Errore hermes_eye: {e}")

    def test_06_filters_logic(self):
        """Verifica hermes_filters.py (FilterLogic)"""
        try:
            from hermes_filters import FilterLogic

            logic = FilterLogic()
            self.assertIsNotNone(logic)
            self.assertIsNone(logic.df)  # Deve partire vuoto
            print("✅ hermes_filters: FilterLogic inizializzato.")
        except Exception as e:
            self.fail(f"Errore hermes_filters: {e}")

    def test_07_stats_logic(self):
        """Verifica hermes_stats.py (StatsLogic)"""
        try:
            from hermes_stats import StatsLogic

            logic = StatsLogic()
            self.assertIsNotNone(logic)
            print("✅ hermes_stats: StatsLogic inizializzato.")
        except Exception as e:
            self.fail(f"Errore hermes_stats: {e}")

    def test_08_reviewer_logic(self):
        """Verifica hermes_reviewer.py (ReviewerLogic)"""
        try:
            from hermes_reviewer import ReviewerLogic

            logic = ReviewerLogic()
            self.assertIsNotNone(logic)
            self.assertEqual(logic.current_frame, 0)
            print("✅ hermes_reviewer: ReviewerLogic inizializzato.")
        except Exception as e:
            self.fail(f"Errore hermes_reviewer: {e}")


if __name__ == "__main__":
    print("--- AVVIO TEST DI VALIDAZIONE LOGICA (HEADLESS) ---")
    unittest.main(verbosity=2)
