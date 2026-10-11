// =====================================================================================
// BRISCOLA CARD DETECTOR & RECOGNIZER
// -------------------------------------------------------------------------------------
// Questo programma:
//   1) Rileva rettangoli "simili a carte" in un'immagine (bordi, forma, area, luminosità)
//   2) Rimuove i rettangoli duplicati/sovrapposti (non-maximum suppression semplificata)
//   3) Raddrizza (rettifica prospetticamente) ogni carta rilevata in un'immagine 300x500
//   4) Confronta ogni carta raddrizzata (anche in versione specchiata) con un set di
//      immagini template tramite differenza assoluta media dei pixel (MAD),
//      per capire quale carta del mazzo sia stata fotografata.
// =====================================================================================

#include <opencv2/highgui.hpp>	 // funzioni di visualizzazione (imshow, waitKey, ecc.)
#include <opencv2/imgcodecs.hpp> // lettura/scrittura immagini (imread)
#include <opencv2/imgproc.hpp>	 // elaborazione immagini (Canny, contorni, trasformazioni, ecc.)

#include <filesystem> // per scorrere la cartella dei template (directory_iterator)
#include <iostream>
#include <limits> // per std::numeric_limits (valore massimo double iniziale)
#include <string>
#include <vector>

namespace fs = std::filesystem; // alias comodo per std::filesystem

// -------------------------------------------------------------------------------------
// Struttura che rappresenta un "template", cioè una carta di riferimento nota:
// memorizza il percorso del file (per poterlo ri-visualizzare/stampare più avanti)
// e l'immagine in scala di grigi già ridimensionata alla dimensione standard.
// -------------------------------------------------------------------------------------
struct TemplateFeatures
{
	fs::path path;	   // percorso del file immagine del template (es. "sette_di_spade.png")
	cv::Mat grayImage; // immagine del template convertita in grigio e ridimensionata
};


int main(int argc, char **argv)
{

	// ---------------------------------------------------------------------------
	// Determina il percorso dell'immagine da analizzare: se viene passato un
	// argomento da riga di comando lo usa, altrimenti usa un percorso di default
	// (utile per test rapidi durante lo sviluppo in Visual Studio).
	// ---------------------------------------------------------------------------
	const std::string imagePath = argc >= 2
									  ? argv[1]
									  : "D:/UNI/LabCV/BRISCOLA/BRISCOLA_project_CV/test image/8.png";

	// Carica l'immagine a colori (BGR, formato standard di OpenCV)
	cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
	if (image.empty())
	{
		std::cerr << "Impossibile aprire l'immagine: " << imagePath << "\n";
		return 1; // termina con codice di errore se l'immagine non è stata caricata
	}

	// =============================================================================
	// FASE 4 — CARICAMENTO DEI TEMPLATE (IMMAGINI DI RIFERIMENTO DELLE CARTE)
	// =============================================================================

	// Dimensione standard a cui verranno ridimensionati sia i template sia le
	// carte rilevate/raddrizzate, per poterle confrontare pixel per pixel
	// (il confronto richiede immagini della stessa identica dimensione).
	const cv::Size templateSize(300, 500);

	// Cartella contenente le immagini di riferimento (una per ogni carta del mazzo,
	// es. "asso_di_spade.png", "sette_di_coppe.png", ecc.)
	const fs::path templateFolder =
		"D:/UNI/LabCV/BRISCOLA/BRISCOLA_project_CV/Briscola_Trentine";

	std::vector<TemplateFeatures> templates; // lista di tutti i template caricati

	// Scorre tutti i file presenti nella cartella dei template
	for (const fs::directory_entry &entry : fs::directory_iterator(templateFolder))
	{
		if (!entry.is_regular_file())
		{
			continue; // salta sottocartelle o file speciali, considera solo file normali
		}

		// Carica l'immagine del template a colori
		cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
		if (img.empty())
		{
			continue; // file non leggibile come immagine: lo ignora
		}

		cv::Mat templateGray;
		// Converte in scala di grigi (il confronto successivo avviene su un solo canale)
		cv::cvtColor(img, templateGray, cv::COLOR_BGR2GRAY);

		// Ridimensiona forzatamente il template alla dimensione standard 300x500,
		// indipendentemente dalla risoluzione/proporzioni originali del file,
		// così da poterlo confrontare pixel-per-pixel con le carte rilevate
		// (anch'esse portate alla stessa dimensione da rectifyRectangle).
		cv::resize(templateGray, templateGray, templateSize);

		// Aggiunge il template (percorso + immagine grigia) alla lista
		templates.push_back({entry.path(), templateGray});
	}

	// =============================================================================
	// FASE 5 — RICONOSCIMENTO: CONFRONTO DI OGNI CARTA RILEVATA CON I TEMPLATE
	// =============================================================================

	// Scorre ogni rettangolo (carta) rilevato e "ripulito" nelle fasi precedenti
	for (size_t i = 0; i < detectedRectangles.size(); ++i)
	{

		// Raddrizza prospetticamente la carta corrente, ottenendo un'immagine
		// a colori 300x500 (o vuota se il rettangolo era troppo piccolo)
		cv::Mat rectified = rectifyRectangle(image, detectedRectangles[i]);
		if (rectified.empty())
		{
			continue; // rettangolo non valido: salta al successivo
		}

		cv::Mat rectifiedGray;
		// Converte la carta raddrizzata in scala di grigi
		cv::cvtColor(rectified, rectifiedGray, cv::COLOR_BGR2GRAY);

		// Ridimensiona nuovamente alla dimensione standard (ridondante rispetto
		// a rectifyRectangle, che produce già 300x500, ma garantisce coerenza
		// assoluta delle dimensioni prima del confronto pixel-per-pixel)
		cv::resize(rectifiedGray, rectifiedGray, templateSize);

		cv::Mat mirroredGray;
		// Crea anche una versione "specchiata orizzontalmente" della carta
		// (flip con codice 1 = ribaltamento lungo l'asse verticale, cioè sinistra-destra).
		// Serve a gestire il caso in cui la carta sia stata rilevata/raddrizzata
		// con un orientamento speculare rispetto al template (es. per via
		// dell'ordine con cui orderPoints assegna gli angoli in situazioni ambigue).
		cv::flip(rectifiedGray, mirroredGray, 1);

		fs::path bestMatch;											// percorso del template con la miglior corrispondenza finora
		double bestDifference = std::numeric_limits<double>::max(); // inizializzata al massimo possibile
		bool bestWasMirrored = false;								// indica se il miglior match è stato trovato usando la versione specchiata

		// Confronta la carta corrente (sia normale sia specchiata) con OGNI template
		for (const TemplateFeatures &templ : templates)
		{

			cv::Mat difference;
			// Calcola la differenza assoluta pixel-per-pixel tra la carta raddrizzata
			// (non specchiata) e il template corrente
			cv::absdiff(rectifiedGray, templ.grayImage, difference);
			// Media di tutte le differenze: un singolo numero che rappresenta
			// quanto le due immagini sono complessivamente diverse (0 = identiche)
			double currentDifference = cv::mean(difference)[0];

			// Se questa è la differenza più bassa trovata finora, aggiorna il miglior match
			if (currentDifference < bestDifference)
			{
				bestDifference = currentDifference;
				bestMatch = templ.path;
				bestWasMirrored = false;
			}

			// Ripete lo stesso confronto usando la versione SPECCHIATA della carta,
			// nel caso in cui l'orientamento speculare dia una corrispondenza migliore
			cv::absdiff(mirroredGray, templ.grayImage, difference);
			currentDifference = cv::mean(difference)[0];

			if (currentDifference < bestDifference)
			{
				bestDifference = currentDifference;
				bestMatch = templ.path;
				bestWasMirrored = true; // il match migliore è arrivato dalla versione specchiata
			}
		}

		// Stampa a console il risultato del riconoscimento per la carta corrente:
		// indice progressivo, nome del file template vincente e valore di differenza
		// (più basso = più simile)
		std::cout << "Riquadro " << i + 1 << "/" << detectedRectangles.size()
				  << ": miglior match = " << bestMatch.filename().string()
				  << " (differenza = " << bestDifference << ").\n";

		// Mostra la versione della carta rilevata che ha prodotto il miglior match
		// (normale o specchiata, a seconda di bestWasMirrored)
		cv::imshow("Riquadro rettificato", bestWasMirrored ? mirroredGray : rectifiedGray);

		// Se è stato trovato un match valido, carica e mostra anche l'immagine
		// originale del template vincente, per confronto visivo
		if (!bestMatch.empty())
		{
			cv::Mat bestImage = cv::imread(bestMatch.string(), cv::IMREAD_COLOR);
			cv::imshow("Template migliore", bestImage);
		}

		// Attende la pressione di un tasto prima di passare alla carta successiva.
		// Se l'utente preme ESC, 'q' o 'Q', interrompe il ciclo e termina prima
		// di aver esaminato tutte le carte rilevate.
		int key = cv::waitKey(0);
		if (key == 27 || key == 'q' || key == 'Q')
		{
			break;
		}
	}

	// Chiude tutte le finestre OpenCV aperte prima di terminare il programma
	cv::destroyAllWindows();

	return 0;
}
