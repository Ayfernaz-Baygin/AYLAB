"""Uygulama giriş noktası: python main.py [görüntü_yolu]"""

import sys

from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("AYLAB")
    app.setStyle("Fusion")  # Yerel stil farklarından bağımsız, tutarlı çizim

    window = MainWindow()
    window.show()

    # İsteğe bağlı: komut satırından verilen görüntüyü doğrudan aç.
    if len(sys.argv) > 1:
        window.load_image(sys.argv[1])

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
