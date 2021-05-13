# Pokyny pro instalaci a spuštění apliakce
## Prerekvizity
Jedinou prerekvizitou pro instalaci programu je nainstalovaná conda. Nejjednodušší způsob, jak ji získat, je instalace Minicondy, což je minimální verze Anacondy, která obsahuje pouze příkaz conda a příslušné závislosti. Instalace pro kteroukoliv distribuci (Windows, Linux, Mac OS) je dostupná na adrese: https://docs.conda.io/en/latest/miniconda.html. 


## Naklonování repozitáře
Nejprve je nutné naklonovat repozitář se zdrojovým kódem a~ ostatními potřebnými soubory. Repozitář je dostupný na adrese: https://github.com/vacenovskyrichard/carcounter. Repozitář je možné stáhnout jako zip nebo naklonovat pomocí příkazu git clone.

## Stažení oficiálního YOLOv4 Tiny weights souboru
Dále je potřeba samostatně stáhnout oficiální předtrénovaný soubor YOLOv4 Tiny weights. Ten je dostupný na následující adrese: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights. Soubor je nutné uložit do složky data.

## Vytvoření a spuštění virtuálního prostředí
Pomocí příkazu conda je nutné vytvořit virtuální prostředí, které obsahuje všechny potřebné knihovny a balíčky. Virtuální prostředí vytvoříme pomocí následujícího příkazu:


      $ conda env create -f conda-carcounter.yml

## Aktivace prostředí:

      $ conda activate car_counter

## Zformátování YOLOv4 Tiny weights souboru do TensorFlow formátu
Pro zformátování slouží následující příkaz:

      $ python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --model yolov4 --tiny


## Spuštění
V tuto chvíli by již mělo být vše připraveno. Pro spuštění celé aplikace slouží příkaz:

      $ python main.py

