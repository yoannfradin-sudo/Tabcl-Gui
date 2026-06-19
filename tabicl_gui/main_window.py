from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QMessageBox, QApplication,
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt

from tabicl_gui.tabs.data_tab import DataTab
from tabicl_gui.tabs.knowledge_tab import KnowledgeTab
from tabicl_gui.tabs.train_tab import TrainTab
from tabicl_gui.tabs.results_tab import ResultsTab
from tabicl_gui.tabs.forecast_tab import ForecastTab
from tabicl_gui.tabs.finetune_tab import FinetuneTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TabICL GUI")
        self.resize(1200, 800)

        # Shared application state
        self._state: dict = {}

        self._build_menu()
        self._build_tabs()

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _build_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("Fichier")
        quit_action = QAction("Quitter", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(QApplication.quit)
        file_menu.addAction(quit_action)

        help_menu = menubar.addMenu("Aide")
        about_action = QAction("À propos", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------

    def _build_tabs(self):
        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        self._data_tab = DataTab(self._state)
        self._knowledge_tab = KnowledgeTab(self._state)
        self._train_tab = TrainTab(self._state)
        self._results_tab = ResultsTab(self._state)
        self._forecast_tab = ForecastTab(self._state)
        self._finetune_tab = FinetuneTab(self._state)

        self._tabs.addTab(self._data_tab, "Données")
        self._tabs.addTab(self._knowledge_tab, "Connaissances (LLM)")
        self._tabs.addTab(self._train_tab, "Entraînement")
        self._tabs.addTab(self._results_tab, "Résultats")
        self._tabs.addTab(self._forecast_tab, "Prévision")
        self._tabs.addTab(self._finetune_tab, "Fine-tuning")

        # Lock non-data tabs until data is loaded
        for i in range(1, self._tabs.count()):
            self._tabs.setTabEnabled(i, False)

        # Wire signals
        self._data_tab.data_confirmed.connect(self._on_data_confirmed)
        self._train_tab.training_done.connect(self._on_training_done)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_data_confirmed(self, state: dict):
        self._state.update(state)

        # Enable all tabs
        for i in range(self._tabs.count()):
            self._tabs.setTabEnabled(i, True)

        # Propagate to tabs that need it
        self._knowledge_tab.on_data_confirmed(state)
        self._train_tab.on_data_confirmed(state)
        self._forecast_tab.on_data_confirmed(state)
        self._finetune_tab.on_data_confirmed(state)

        # Navigate to the knowledge tab (étape métier avant l'entraînement)
        self._tabs.setCurrentIndex(1)

    def _on_training_done(self, state: dict):
        self._state.update(state)
        self._results_tab.on_training_done(state)
        self._tabs.setCurrentIndex(3)   # onglet Résultats

    # ------------------------------------------------------------------
    # About
    # ------------------------------------------------------------------

    def _show_about(self):
        QMessageBox.about(
            self,
            "À propos de TabICL GUI",
            "<b>TabICL GUI</b><br>"
            "Interface graphique pour <a href='https://github.com/soda-inria/tabicl'>TabICL</a>, "
            "un modèle de fondation tabulaire basé sur l'apprentissage en contexte.<br><br>"
            "Fonctionnalités : Classification, Régression, Prévision temporelle, "
            "SHAP, Fine-tuning.",
        )
