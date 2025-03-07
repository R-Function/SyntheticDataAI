# Image Caption RNN
Installation:
Bsp.: Conda
- conda create --name _name_
- conda activate _name_
- conda install pip
- pip install -r requirements.txt

Benutzung:
Das Programm ist über das Terminal aufzurufen
- mit dem Terminal ins verzeichnis navigieren.
- python3 ./main.py -h
    - Gibt Infos zur Benutzung des Programms aus.
- python3 ./main.py [config_datei] 
    - Das Programm erwartet eine Ausgefüllte .json Datei mit den entsprechenden Konfigurationen. Mehr Infors unten.
    - Ohne weitere Argumente werden alle Programmteile in Reihenfolge ausgeführt.
- python3 ./main.py [config_datei] --execute
    - Mit diesem Argument kann festgelegt werden, welcher Teil des Programms ausgeführt werden sollte.
    - mit _prep-data_ werden die Trainingsdaten vorbearbeitet und in .pkl Dateien überführt.
    - mit _train_ wird das Training durchgeführt. Hyperparameter können in der Konfig festgelegt werden.
    - mit _execute-model_ kann man testweise ein Bild von einem fertig trainiertem RNN beschreiben lassen. Über Beam Search werden die besten ergebnisse rausgesucht. Der Model- und Bildpfad, sowie die Beam-Search einstellungen (beam_search_width, caption_max_length, vocab_size) können über die config.json eingestellt werden.
    - mit _evaluate_ werden BLEU und ROUGE Score ermittelt.

Die Trainingsdaten:
Trainingsdaten müssen wie folgt strukturiert sein:
- Zwei Ordner, einer mit Bildern und der andere mit .txt dateien
- Die Bilder sollten im .jpg Format sein
- Der Ordner mit den Textdateien beinhaltet:
    - eine Datei mit den Bildbeschreibungen. Es sind mehrere Beschreibungen pro Bild Möglich. Das Format muss aber wie im Beispiel gezeigt sein.
    - Drei Dateien mit den Dateinamen der Bilder im Datenset. Von den drei Datein soll je eine die Trainings-, Test- und Evaluationsbilder spezifizieren.

Bsp. Bildbeschreibung:
_bildpfad_#_caption-index_ _beschreibung_
95151149_5ca6747df6.jpg#1	A mountainous photo is complete with a blue sky .
95151149_5ca6747df6.jpg#2	A snowy mountain range .
95151149_5ca6747df6.jpg#3	Rocky mountains .
952171414_2db16f846f.jpg#0	a group of four people conversing next to a bus
952171414_2db16f846f.jpg#1	Four people taking in front of a bus . 

Die Einstellungen in der Konfigdatei:
Die Config.Json ist wie folgt aufgebaut. Die einzelnen Werte werden hier in ihrer benutzung beschrieben.
"data files and directories":
    "token_path"          : Dateipfad der Annotationstextdatei
    "train_set_path"      : Dateipfad der .txt, die die Trainingsbilder festlegt
    "dev_set_path"        : Dateipfad der .txt, die die Trainingstestbilder festlegt
    "test_set_path"       : Dateipfad der .txt die die Evaluationbilder für das fertigtrainierte Modell festlegt
    "embedd_path"         : Dateipfad der Wordembeddingdatei
    "data_dir"            : Das Verzeichnis, in dem die Bilder des Datensets enthalten sind. 
    "dest_dir"            : Das Verzeichnis, in dem das Model abgespeichert werden soll.
    "model_path"          : Der Dateipfad eines fertig trainerten Modells, das getestet werden soll.
    "test_img_path"       : Der Dateipfad eines Bildes, welches durch das Modell beschrieben werden soll.

"hyperparams":{
    "vocab_size"          : 
    "caption_max_length"  : 
    "batch_size"          : 
    "epochs"              : 
    "beam_search_width"   : 
    "embedding_dim"       :  
}