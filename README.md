# Image Caption RNN

## Installation

Beispiel mit Conda:

```bash
conda create --name <env_name>
conda activate <env_name>
conda install pip
pip install -r requirements.txt
```

## Benutzung

Das Programm wird über das Terminal ausgeführt:

1. Navigiere im Terminal ins Projektverzeichnis.
2. Rufe die Hilfe auf:
   ```bash
   python3 ./main.py -h
   ```
   Dies gibt eine Übersicht zur Benutzung des Programms.
3. Starte das Programm mit einer Konfigurationsdatei:
   ```bash
   python3 ./main.py <config_datei>
   ```
   - Erwartet eine ausgefüllte `.json`-Datei mit den entsprechenden Konfigurationen (siehe unten).
   - Ohne zusätzliche Argumente werden alle Programmteile in der richtigen Reihenfolge ausgeführt.
4. Teile des Programms gezielt ausführen:
   ```bash
   python3 ./main.py <config_datei> --execute <task>
   ```
   - `prep-data`: Vorverarbeitung der Trainingsdaten und Speicherung als `.pkl`-Dateien.
   - `train`: Startet das Training. Hyperparameter sind in der Konfigurationsdatei definiert.
   - `execute-model`: Lässt ein trainiertes RNN ein Bild beschreiben. Beam Search findet die besten Ergebnisse.
   - `evaluate`: Berechnet BLEU- und ROUGE-Scores zur Bewertung des Modells.

## Trainingsdaten

Die Daten müssen wie folgt strukturiert sein:

- **Bilderordner**: Enthält `.jpg`-Bilder.
- **Textordner**: Enthält:
  - Eine Datei mit Bildbeschreibungen (mehrere Beschreibungen pro Bild möglich).
  - Drei Dateien mit den Dateinamen der Bilder für Training, Test und Evaluation.

### Beispiel Bildbeschreibungsdatei:
```
95151149_5ca6747df6.jpg#1 A mountainous photo is complete with a blue sky.
95151149_5ca6747df6.jpg#2 A snowy mountain range.
95151149_5ca6747df6.jpg#3 Rocky mountains.
952171414_2db16f846f.jpg#0 A group of four people conversing next to a bus.
952171414_2db16f846f.jpg#1 Four people talking in front of a bus.
```

## Konfigurationsdatei (`config.json`)

Die `config.json` enthält die folgenden Parameter:

```json
{
    "data files and directories": {
        "token_path": "Pfad zur Annotationstextdatei",
        "train_set_path": "Pfad zur Datei, die die Trainingsbilder festlegt",
        "dev_set_path": "Pfad zur Datei, die die Trainings-Testbilder festlegt",
        "test_set_path": "Pfad zur Datei, die die Evaluationsbilder für das fertig trainierte Modell festlegt",
        "embedd_path": "Pfad zur Word-Embedding-Datei",
        "data_dir": "Verzeichnis, in dem die Bilder des Datensets enthalten sind",
        "dest_dir": "Verzeichnis, in dem das Modell gespeichert werden soll",
        "model_path": "Pfad zu einem fertig trainierten Modell, das getestet werden soll",
        "test_img_path": "Pfad zu einem Bild, das durch das Modell beschrieben werden soll"
    },
    "hyperparams": {
        "vocab_size": "Größe des Wortschatzes",
        "caption_max_length": "Maximale Länge einer Bildbeschreibung",
        "batch_size": "Anzahl der Trainingsbeispiele pro Batch",
        "epochs": "Anzahl der Trainingsdurchläufe (Epochen)",
        "beam_search_width": "Anzahl der Strahlen im Beam Search Algorithmus",
        "embedding_dim": "Dimension der Wort-Embeddings"
    }
}
```

---

## Lizenz
Dieses Projekt steht unter der MIT-Lizenz. Weitere Informationen in der `LICENSE`-Datei.
