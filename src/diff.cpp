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

// =====================================================================================
// orderPoints
// -------------------------------------------------------------------------------------
// Dato un poligono a 4 vertici (in un ordine NON garantito, così come esce da
// approxPolyDP), riordina i punti secondo la convenzione:
//   ordered[0] = angolo in alto a sinistra   (top-left)
//   ordered[1] = angolo in alto a destra     (top-right)
//   ordered[2] = angolo in basso a destra    (bottom-right)
//   ordered[3] = angolo in basso a sinistra  (bottom-left)
//
// Il trucco geometrico usato:
//   - il punto con SOMMA (x+y) più piccola è quello più vicino all'origine (0,0),
//     quindi è l'angolo in alto a sinistra.
//   - il punto con SOMMA (x+y) più grande è il più lontano dall'origine,
//     quindi è l'angolo in basso a destra.
//   - il punto con DIFFERENZA (x-y) più piccola (cioè y grande rispetto a x)
//     è l'angolo in alto a destra.
//   - il punto con DIFFERENZA (x-y) più grande (cioè x grande rispetto a y)
//     è l'angolo in basso a sinistra.
// Questo funziona perché in un sistema immagine (y cresce verso il basso) queste
// combinazioni separano in modo affidabile i 4 angoli di un quadrilatero convesso.
// =====================================================================================
static std::vector<cv::Point2f> orderPoints(const std::vector<cv::Point> &polygon)
{
	// Converte i punti interi (cv::Point) in punti float (cv::Point2f),
	// necessari per i calcoli geometrici successivi (getPerspectiveTransform ecc.)
	std::vector<cv::Point2f> points;
	for (const cv::Point &point : polygon)
	{
		points.emplace_back(static_cast<float>(point.x), static_cast<float>(point.y));
	}

	std::vector<cv::Point2f> ordered(4); // vettore di output già dimensionato a 4 elementi

	// Inizializza i valori estremi (somma/differenza min e max) usando il primo punto
	float minSum = points[0].x + points[0].y;
	float maxSum = minSum;
	float minDiff = points[0].x - points[0].y;
	float maxDiff = minDiff;

	// Indici (nel vettore points) dei 4 angoli individuati finora
	int topLeft = 0;
	int bottomRight = 0;
	int topRight = 0;
	int bottomLeft = 0;

	// Scorre i restanti 3 punti (indice 1,2,3) aggiornando i 4 estremi
	for (int i = 1; i < 4; ++i)
	{
		float sum = points[i].x + points[i].y;
		float diff = points[i].x - points[i].y;

		if (sum < minSum)
		{ // somma minima trovata -> nuovo candidato top-left
			minSum = sum;
			topLeft = i;
		}
		if (sum > maxSum)
		{ // somma massima trovata -> nuovo candidato bottom-right
			maxSum = sum;
			bottomRight = i;
		}
		if (diff < minDiff)
		{ // differenza minima -> nuovo candidato top-right
			minDiff = diff;
			topRight = i;
		}
		if (diff > maxDiff)
		{ // differenza massima -> nuovo candidato bottom-left
			maxDiff = diff;
			bottomLeft = i;
		}
	}

	// Assegna i punti ordinati nella convenzione TL, TR, BR, BL
	ordered[0] = points[topLeft];
	ordered[1] = points[topRight];
	ordered[2] = points[bottomRight];
	ordered[3] = points[bottomLeft];
	return ordered;
}

// =====================================================================================
// rectifyRectangle
// -------------------------------------------------------------------------------------
// Prende l'immagine originale e un poligono a 4 vertici (il contorno di una carta
// rilevata, ancora prospetticamente distorto) e restituisce un'immagine "raddrizzata"
// di dimensione fissa 300x500, come se la carta fosse fotografata perfettamente
// frontale. Questo si ottiene calcolando un'omografia (trasformazione prospettica)
// tra i 4 angoli rilevati e i 4 angoli di un rettangolo "ideale" di output.
// =====================================================================================
static cv::Mat rectifyRectangle(const cv::Mat &image, const std::vector<cv::Point> &polygon)
{
	// Riordina i vertici del poligono nella convenzione TL, TR, BR, BL
	std::vector<cv::Point2f> source = orderPoints(polygon);

	// Stima la larghezza del rettangolo come il massimo tra:
	//  - la distanza tra angolo TR e TL (lato superiore)
	//  - la distanza tra angolo BR e BL (lato inferiore)
	// Si prende il massimo per essere robusti a piccole distorsioni prospettiche
	// (i due lati "paralleli" potrebbero non avere esattamente la stessa lunghezza
	// nell'immagine a causa della prospettiva).
	float width = std::max(cv::norm(source[1] - source[0]), cv::norm(source[2] - source[3]));

	// Analogamente, stima l'altezza come il massimo tra il lato sinistro e quello destro
	float height = std::max(cv::norm(source[3] - source[0]), cv::norm(source[2] - source[1]));

	// Se il rettangolo rilevato è troppo piccolo (meno di 10 pixel di lato),
	// probabilmente è rumore: restituisce una Mat vuota per segnalare il fallimento
	if (width < 10.0f || height < 10.0f)
	{
		return {};
	}

	// Dimensioni fisse dell'immagine di output: tutte le carte, indipendentemente
	// dalla loro dimensione/distanza nella foto originale, vengono normalizzate
	// a questa stessa risoluzione, per poter essere confrontate in modo omogeneo
	// con i template più avanti.
	constexpr int outputWidth = 300;
	constexpr int outputHeight = 500;

	// Punti di destinazione "di default": assume che il rettangolo rilevato sia
	// già verticale (più alto che largo), quindi mappa:
	//   TL -> (0,0)                         angolo alto-sinistra dell'output
	//   TR -> (outputWidth-1, 0)             angolo alto-destra
	//   BR -> (outputWidth-1, outputHeight-1) angolo basso-destra
	//   BL -> (0, outputHeight-1)            angolo basso-sinistra
	std::vector<cv::Point2f> destination = {
		{0.0f, 0.0f},
		{static_cast<float>(outputWidth - 1), 0.0f},
		{static_cast<float>(outputWidth - 1), static_cast<float>(outputHeight - 1)},
		{0.0f, static_cast<float>(outputHeight - 1)}};

	// Se il rettangolo e' orizzontale, ruota la destinazione per mostrare
	// sempre la carta in formato verticale.
	//
	// Caso: la carta è stata rilevata "sdraiata" (più larga che alta) nella foto.
	// Per ottenere comunque un'immagine di output verticale (300 larghezza x 500 altezza,
	// cioè outputWidth x outputHeight), si scambiano i ruoli: i punti sorgente TL,TR,BR,BL
	// vengono mappati su una destinazione di dimensione (outputHeight x outputWidth),
	// cioè effettivamente si "ruota" la carta di 90° durante il raddrizzamento.
	if (width > height)
	{
		destination = {
			{0.0f, 0.0f},
			{static_cast<float>(outputHeight - 1), 0.0f},
			{static_cast<float>(outputHeight - 1), static_cast<float>(outputWidth - 1)},
			{0.0f, static_cast<float>(outputWidth - 1)}};

		// Calcola la matrice di trasformazione prospettica 3x3 che mappa i 4 punti
		// sorgente (nell'immagine originale) sui 4 punti destinazione appena definiti
		cv::Mat transform = cv::getPerspectiveTransform(source, destination);

		cv::Mat warped;
		// Applica la trasformazione: warpPerspective deforma l'immagine originale
		// secondo la matrice 'transform', producendo un output di dimensione
		// (outputHeight x outputWidth) = (500 x 300) in questo ramo "ruotato"
		cv::warpPerspective(image, warped, transform, cv::Size(outputHeight, outputWidth));
		return warped;
	}

	// Caso "normale": il rettangolo è già più alto che largo (o quadrato).
	// Calcola la trasformazione prospettica verso la destinazione di default
	// (300 larghezza x 500 altezza) e la applica.
	cv::Mat transform = cv::getPerspectiveTransform(source, destination);
	cv::Mat warped;
	cv::warpPerspective(image, warped, transform, cv::Size(outputWidth, outputHeight));
	return warped;
}

// =====================================================================================
// isCardLikeRectangle
// -------------------------------------------------------------------------------------
// Funzione filtro: decide se un poligono (ottenuto da approxPolyDP su un contorno)
// "assomiglia" a una carta da gioco, applicando una serie di controlli euristici
// su forma, area e luminosità. Restituisce true solo se TUTTI i controlli passano.
// =====================================================================================
static bool isCardLikeRectangle(const std::vector<cv::Point> &polygon, double area, double imageArea,
								const cv::Mat &gray, const cv::Rect &bounds)
{

	// CONTROLLO 1: il poligono deve avere esattamente 4 vertici (quadrilatero)
	// ed essere convesso (nessun angolo "rientrante"). Scarta forme irregolari,
	// contorni frammentati o poligoni con più/meno di 4 lati.
	if (polygon.size() != 4 || !cv::isContourConvex(polygon))
	{
		return false;
	}

	// Calcola il rettangolo minimo (anche ruotato) che racchiude il poligono.
	// A differenza di boundingRect (che è sempre allineato agli assi),
	// minAreaRect tiene conto anche della rotazione, dando lati più accurati
	// per un rettangolo inclinato.
	cv::RotatedRect rectangle = cv::minAreaRect(polygon);
	float longSide = std::max(rectangle.size.width, rectangle.size.height);	 // lato lungo
	float shortSide = std::min(rectangle.size.width, rectangle.size.height); // lato corto

	// CONTROLLO 2: rapporto tra lato corto e lato lungo. Le carte da gioco hanno
	// una proporzione tipica (non sono né quadrate né strisce sottilissime).
	// Si scarta se il lato lungo è nullo (rettangolo degenere) o se il rapporto
	// shortSide/longSide è minore di 0.4 (forma troppo allungata per essere una carta,
	// probabilmente un'ombra o un bordo del tavolo).
	if (longSide <= 0.0f || shortSide / longSide < 0.4f)
	{
		return false;
	}

	// CONTROLLO 3: area plausibile rispetto all'area totale dell'immagine.
	// Scarta rettangoli troppo piccoli (< 0.2% dell'immagine, probabile rumore/frammenti)
	// o troppo grandi (> 80% dell'immagine, probabilmente l'intero tavolo/sfondo
	// piuttosto che una singola carta).
	if (area < 0.002 * imageArea || area > 0.8 * imageArea)
	{
		return false;
	}

	// Calcola l'intersezione tra il bounding box del poligono e i limiti reali
	// dell'immagine, per evitare di leggere pixel fuori dai bordi (che causerebbe
	// un errore/crash in caso il bounding box uscisse leggermente dall'immagine
	// per via delle approssimazioni di approxPolyDP).
	cv::Rect safeBounds = bounds & cv::Rect(0, 0, gray.cols, gray.rows);
	if (safeBounds.empty())
	{
		return false; // bounding box non valido (nessuna intersezione con l'immagine)
	}

	// Il retro della briscola e' molto scuro e contiene una texture fitta.
	// Le carte scoperte hanno invece una superficie chiara: escludiamo il
	// retro senza scartare i rettangoli chiari trovati tra la tovaglia.
	//
	// CONTROLLO 4: calcola la luminosità media dei pixel in scala di grigi
	// all'interno del bounding box. Se è troppo bassa (< 90 su una scala 0-255),
	// si presume che si tratti del dorso scuro di una carta coperta (o di un'ombra),
	// e viene scartato: si vogliono riconoscere solo le carte scoperte (fronte).
	double meanBrightness = cv::mean(gray(safeBounds))[0];
	return meanBrightness >= 120.0;
}

// =====================================================================================
// MAIN
// =====================================================================================
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
	// FASE 1 — PRE-ELABORAZIONE E RILEVAMENTO DEI BORDI
	// =============================================================================

	cv::Mat gray;
	// Conversione in scala di grigi: semplifica l'elaborazione successiva
	// (edge detection, calcolo luminosità) lavorando su un solo canale.
	cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

	// Sfocatura Gaussiana (kernel 5x5) per attenuare rumore ad alta frequenza
	// che altrimenti genererebbe falsi bordi nel passo di Canny successivo.
	cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0.0);

	cv::Mat edges;
	// Canny edge detector: individua i contorni netti nell'immagine.
	// I due parametri (25.0, 90.0) sono le soglie basse/alte dell'algoritmo
	// di isteresi di Canny: bordi con gradiente sopra 90 sono sempre considerati
	// bordi "forti"; quelli tra 25 e 90 sono inclusi solo se collegati a un bordo forte.
	cv::Canny(gray, edges, 25.0, 90.0);

	cv::Mat closedEdges;
	// Kernel rettangolare 3x3 per l'operazione morfologica successiva
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

	// Chiusura morfologica (dilatazione seguita da erosione): serve a "saldare"
	// piccole interruzioni/discontinuità nei bordi rilevati da Canny, così che
	// il contorno di una carta risulti una linea chiusa e continua (necessario
	// per un corretto findContours più avanti).
	cv::morphologyEx(edges, closedEdges, cv::MORPH_CLOSE, kernel);

	std::vector<std::vector<cv::Point>> contours;
	// Estrae tutti i contorni dall'immagine binaria dei bordi.
	// RETR_LIST: recupera tutti i contorni senza costruire gerarchie annidate
	//            (non interessa sapere quali contorni sono "dentro" altri).
	// CHAIN_APPROX_SIMPLE: comprime i segmenti orizzontali/verticali/diagonali
	//            mantenendo solo i punti finali, riducendo la memoria usata.
	cv::findContours(closedEdges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

	// =============================================================================
	// FASE 2 — FILTRAGGIO DEI CONTORNI "SIMILI A CARTE"
	// =============================================================================

	// Maschera binaria (stessa dimensione dell'immagine originale, 1 canale):
	// verrà "accesa" (255) solo nelle zone corrispondenti alle carte rilevate.
	cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);

	// Area totale dell'immagine, usata come riferimento per il filtro sull'area
	// relativa dei rettangoli candidati (vedi isCardLikeRectangle).
	double imageArea = static_cast<double>(image.cols) * image.rows;

	int rectangles = 0;										// contatore finale dei rettangoli validi (aggiornato più avanti)
	std::vector<std::vector<cv::Point>> detectedRectangles; // poligoni accettati come "carte"

	// Scorre tutti i contorni individuati da findContours
	for (const auto &contour : contours)
	{
		double area = cv::contourArea(contour);			 // area racchiusa dal contorno
		double perimeter = cv::arcLength(contour, true); // perimetro (contorno chiuso: true)

		std::vector<cv::Point> polygon;
		// Approssima il contorno (che può avere molti punti irregolari) con un
		// poligono semplificato. La tolleranza (epsilon) è proporzionale al
		// perimetro (2%): più il contorno è grande, più punti di scostamento
		// sono tollerati. L'obiettivo è ottenere, per i contorni rettangolari,
		// esattamente 4 vertici.
		cv::approxPolyDP(contour, polygon, 0.02 * perimeter, true);

		// Rettangolo di bounding (allineato agli assi) del poligono approssimato,
		// usato per calcolare la luminosità media nella funzione di filtro.
		cv::Rect bounds = cv::boundingRect(polygon);

		// Applica tutti i controlli euristici (forma, area, luminosità):
		// se il poligono non li supera, viene scartato e si passa al contorno successivo.
		if (!isCardLikeRectangle(polygon, area, imageArea, gray, bounds))
		{
			continue;
		}

		// Il poligono ha superato tutti i filtri: viene aggiunto alla lista
		// dei rettangoli rilevati (la maschera verrà disegnata più avanti,
		// DOPO la rimozione dei duplicati).
		detectedRectangles.push_back(polygon);
	}

	// =============================================================================
	// FASE 3 — RIMOZIONE DEI RETTANGOLI DUPLICATI/SOVRAPPOSTI
	// -----------------------------------------------------------------------------
	// Con Canny + approxPolyDP è comune ottenere PIÙ contorni leggermente diversi
	// per lo stesso oggetto reale (es. un contorno "interno" e uno "esterno" dello
	// stesso bordo di una carta, oppure duplicati dovuti a rumore). Questo blocco
	// implementa una semplice forma di "non-maximum suppression" basata sull'IoU
	// (Intersection over Union, qui approssimata come rapporto tra area di
	// sovrapposizione e area del rettangolo più piccolo) per tenere un solo
	// rettangolo per ogni carta reale.
	// =============================================================================
	std::vector<std::vector<cv::Point>> filteredRectangles; // lista finale "pulita"

	// Scorre tutti i rettangoli candidati rilevati nella fase precedente
	for (const auto &rectangle : detectedRectangles)
	{
		// Bounding box (allineato agli assi) del rettangolo corrente,
		// usato per il calcolo rapido della sovrapposizione
		cv::Rect currentBounds = cv::boundingRect(rectangle);
		bool keep = true; // per default si assume che il rettangolo vada tenuto

		// Confronta il rettangolo corrente con tutti quelli già accettati
		// in filteredRectangles (iteratore esplicito perché si può cancellare
		// un elemento durante il ciclo)
		for (auto it = filteredRectangles.begin(); it != filteredRectangles.end();)
		{
			cv::Rect otherBounds = cv::boundingRect(*it);

			// Area di intersezione (sovrapposizione) tra i due bounding box.
			// L'operatore & tra due cv::Rect restituisce il rettangolo intersezione
			// (vuoto se non si sovrappongono).
			cv::Rect overlap = currentBounds & otherBounds;

			// Area del più piccolo tra i due rettangoli: usata come denominatore
			// per normalizzare la sovrapposizione (invece della classica IoU che
			// userebbe l'unione, qui si usa il rettangolo più piccolo: questo rende
			// il criterio più sensibile ai casi in cui un rettangolo piccolo è quasi
			// interamente contenuto in uno più grande, tipico di contorni "doppi").
			double smallerArea = static_cast<double>(std::min(currentBounds.area(), otherBounds.area()));
			double overlapRatio = smallerArea > 0.0 ? overlap.area() / smallerArea : 0.0;

			// Se la sovrapposizione supera l'85% dell'area del rettangolo più piccolo,
			// i due rettangoli sono considerati duplicati della stessa carta reale.
			if (overlapRatio > 0.85)
			{
				if (currentBounds.area() > otherBounds.area())
				{
					// Il rettangolo corrente è più grande di quello già presente in
					// filteredRectangles: si preferisce tenere il rettangolo più grande
					// (probabilmente cattura meglio l'intero bordo della carta), quindi
					// si rimuove quello più piccolo già presente in lista...
					it = filteredRectangles.erase(it);
					continue; // continua a controllare gli eventuali altri duplicati
				}
				// Il rettangolo già presente è più grande (o uguale): si scarta
				// il rettangolo corrente e si interrompe il confronto
				keep = false;
				break;
			}
			++it; // nessuna sovrapposizione significativa: passa al prossimo confronto
		}

		// Se dopo tutti i confronti il rettangolo corrente non è stato scartato,
		// viene aggiunto alla lista finale
		if (keep)
		{
			filteredRectangles.push_back(rectangle);
		}
	}

	// Sostituisce la lista originale (con duplicati) con quella filtrata
	detectedRectangles = filteredRectangles;

	// Ora che i duplicati sono stati rimossi, disegna sulla maschera SOLO
	// i rettangoli definitivi (un poligono pieno per ciascuna carta rilevata)
	for (const auto &rectangle : detectedRectangles)
	{
		cv::fillConvexPoly(mask, rectangle, cv::Scalar(255));
	}
	rectangles = static_cast<int>(detectedRectangles.size()); // conteggio finale

	// Applica la maschera all'immagine originale: maskedImage conterrà i pixel
	// originali solo nelle zone corrispondenti alle carte rilevate, nero altrove.
	// (image.copyTo con maschera copia solo dove la maschera è diversa da zero)
	cv::Mat maskedImage;
	image.copyTo(maskedImage, mask);

	std::cout << "Rettangoli trovati: " << rectangles << "\n"
			  << "Visualizzazione mask e immagine maskata avviata.\n";

	// Mostra a schermo la maschera binaria e l'immagine mascherata,
	// per verificare visivamente la qualità del rilevamento prima di procedere
	cv::imshow("Mask", mask);
	cv::imshow("Immagine maskata", maskedImage);
	cv::waitKey(0); // attende la pressione di un tasto qualsiasi per continuare

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
