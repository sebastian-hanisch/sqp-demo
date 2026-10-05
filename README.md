# SQP – Sequential Quadratic Programming – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-sqp-demo.streamlit.app/)**

Stück 6 der **Nichtlineare-Optimierung-Reihe** der "Konzepte"-Reihe im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Stück 5 löste die restringierte Zylinder-Aufgabe über eine **Folge unrestringierter Probleme**
(Straf-/Barriere). **SQP** (Sequential Quadratic Programming, Wilson 1963) löst sie **direkt**:
an jedem Punkt wird eine kleine quadratische Näherung (QP) des Originalproblems gelöst, deren
Lösung der nächste Schritt ist.

**Einordnung in die Reihe:**

```
Gradientenabstieg (WURZEL)                       [gebaut]
 └─ Newton-Verfahren                             [gebaut]
      └─ Quasi-Newton (BFGS/L-BFGS)               [gebaut]
           └─ Lagrange-Multiplikatoren/KKT        [gebaut]
                ├─ Straf-/Barriere-Verfahren      [gebaut]
                └─ SQP                            [DIESES STÜCK]
                     └─ Innere-Punkte-Verfahren   [gebaut: interior-point-nlp-demo]
 └─ Stochastische Gradientenverfahren             [gebaut: stochastische-gradientenverfahren-demo, letztes Stück]
```

**Vehikel B (wiederverwendet, eigene Kopie ohne Import):** derselbe zylindrische
Transportbehälter — Materialkosten minimieren bei festem Volumen und einem Höhenlimit.

**Zentrale, exakte theoretische Erkenntnis:** SQP mit der EXAKTEN Hesse-Matrix der
Lagrange-Funktion ist mathematisch **identisch** mit Newtons Verfahren auf dem KKT-System aus
Stück 4 (Nocedal & Wright, Kap. 18) — keine Näherung, eine exakte Äquivalenz, hier numerisch
verifiziert (Abweichung $4{,}4\cdot10^{-16}$).

**Ergebnis in Kürze:** SQP konvergiert für beide Presets in **5 Iterationen** auf
Maschinengenauigkeit — dramatisch weniger als die 12 äußeren Stufen, die Straf-/Barriere-Verfahren
(Stück 5) für einen viel gröberen Fehler ($10^{-6}$) brauchen. An einem eigenen, rein linear
restringierten Hilfsproblem konvergiert SQP wie erwartet in **genau einer** Iteration,
unabhängig vom Startpunkt. **Ehrlicher Nebenbefund:** ein neutraler Startwert ($\lambda_0=0$)
verursacht einen vorübergehenden Fehlgriff bei der aktiven Menge (13 statt 5 Iterationen), und von
manchen Startpunkten konvergiert SQP zu einem mathematisch gültigen, aber physikalisch unsinnigen
KKT-Punkt mit negativem Radius — dieselbe Klasse Fund wie Stück 4s SciPy-Fallstrick, hier über
eine völlig andere Methode reproduziert.

## Warum dieses Problem

Stück 5 zeigte, dass man restringierte Probleme über eine Folge unrestringierter Hilfsprobleme
lösen kann — mit dem Preis wachsender (oder wenigstens problembezogener) Ill-Conditioning. SQP
zeigt die historisch erste **direkte** Alternative: keine künstliche Parameterfolge, sondern eine
Folge kleiner, exakt lösbarer quadratischer Modelle des Originalproblems selbst.

## Vorab-Hypothesen (vor der Messung notiert, hier geprüft)

| Hypothese | Ergebnis |
|---|---|
| Bei fixierter aktiver Menge liefert der SQP-Schritt exakt denselben Schritt wie Newton auf dem KKT-System (Stück 4) | ✅ Abweichung $4{,}4\cdot10^{-16}$ |
| SQP konvergiert für beide Presets zur selben Referenzlösung wie Stück 4/5 | ✅ 5 Iterationen, Fehler $\le1{,}8\cdot10^{-15}$ |
| Am rein linear restringierten Hilfsproblem konvergiert SQP in genau 1 Iteration, unabhängig vom Startpunkt | ✅ 4/4 getestete Startpunkte |
| SQP braucht deutlich weniger äußere Iterationen als Straf-/Barriere-Verfahren | ✅ 5 vs. 12 (Straf/Barriere nur bis zum Fehler $10^{-6}$, SQP bis $\approx10^{-15}$) |
| Gradienten-Check gegen finite Differenzen unter $10^{-6}$ | ✅ $2{,}6\cdot10^{-10}$ / $5{,}4\cdot10^{-11}$ |
| ⚠️ **Nicht vorab vermutet, aber gefunden:** ein neutraler Startwert $\lambda_0=0$ kann zu einem vorübergehenden Fehlgriff bei der aktiven Menge führen | ⚠️ bestätigt — 13 statt 5 Iterationen, ein Fall-Wechsel während der Folge |
| ⚠️ **Ebenfalls ungeplant gefunden:** manche Startpunkte konvergieren zu einem physikalisch unsinnigen, aber mathematisch gültigen KKT-Punkt (negativer Radius) | ⚠️ bestätigt — dieselbe Klasse Fund wie Stück 4s SciPy-Fallstrick |

## Befunde (gemessen, keine Behauptungen)

**Exakte Newton-KKT-Reduktion** (Testpunkt $x=(1{,}3,\,2{,}1)$, $\lambda=0{,}5$, Gleichungsfall):
max. Abweichung SQP-Schritt vs. ungedämpfter Newton-Schritt auf dem KKT-System:
$4{,}4\cdot10^{-16}$ (Maschinengenauigkeit).

**Referenz-Konvergenz** (informierter Start $x_0=(V_0^{1/3},V_0^{1/3})$,
$\lambda_0=-2/V_0^{1/3}$ — identisch zu Stück 4):

| $h_{\max}$ | Fall | Iterationen | Fehler |
|---|---|---|---|
| 5,0 | inaktiv | 5 | $1{,}8\cdot10^{-15}$ |
| 1,5 | aktiv | 5 | 0,0 |

**Robustheit über Volumina** $V_0\in\{0{,}5,1,5,10,50,100\}$: 6 von 6 treffen die analytische
Lösung, alle in $\le5$ Iterationen.

**Rein linear restringiertes Hilfsproblem** (kein Vehikel B): $\min(x-2)^2+(y-3)^2$ unter $x+y=4$
— 4 von 4 getesteten Startpunkten ($(0,0)$, $(10,-5)$, $(-3,8)$, $(100,100)$) konvergieren in
**genau einer** Iteration zu $x^*=(1{,}5,\,2{,}5)$.

**Iterationszahl-Vergleich** ($V_0=10$, $h_{\max}=5$, Zielfehler $<10^{-6}$): SQP **5**
Iterationen (auf Maschinengenauigkeit, weit unter dem Zielfehler) gegen **12** äußere Stufen für
sowohl das Straf- als auch das Barriere-Verfahren (Stück 5), um überhaupt erst den viel gröberen
Zielfehler zu erreichen.

**Grenze des Vergleichs:** Für kleine Volumina ($V_0 \lesssim 6$, bei $h_{\max}=5$ gemessen) erreicht das
Strafverfahren (Start $r=h=V_0^{1/3}$) den Zielfehler in den erlaubten 40 äußeren Stufen **gar nicht**: es
bleibt am trivialen Stationärpunkt $r=h=0$ hängen (dort verschwinden Gradient der Oberfläche und der
Volumen-Nebenbedingung, für jedes $\rho$). Die App sagt das dann ausdrücklich und nennt keine Stufenzahl;
das Barriere-Verfahren braucht dort 12 Stufen. Über das gesamte Reglerraster ($V_0=1,4,\dots,100$, $h_{\max}=0{,}5,1,\dots,6$) braucht das Strafverfahren 10–25 Stufen (in 24 von 408 Fällen gar nicht), das Barriere-Verfahren 10–17 Stufen, SQP 3–6 Iterationen.

**Ehrlicher Nebenbefund — neutraler Start scheitert (vorübergehend):** von $\lambda_0=0$ (statt
$-2/V_0^{1/3}$) braucht SQP 13 statt 5 Iterationen; die aktive Menge wechselt während der Folge
mehrfach (inaktiv→aktiv→inaktiv), bevor sie sich stabilisiert. Von Startpunkt $(0{,}5,\,0{,}5)$
konvergiert SQP zu $r=-0{,}7979$ (Fall: aktiv, $h=5{,}0$) — ein mathematisch gültiger KKT-Punkt
(Stationaritäts-Residuum $\approx10^{-13}$, $\mu\ge0$), aber physikalisch unsinnig (negativer
Radius). Dieselbe Ursache wie bei Stück 4s SciPy-Fallstrick: die KKT-Bedingungen charakterisieren
JEDEN stationären Punkt der Lagrange-Funktion, nicht nur den physikalisch sinnvollen — hier über
eine völlig andere Lösungsmethode (Newton auf der Linearisierung statt SLSQP) reproduziert.

## Modell und Verfahren

- `sqp_functions.py` – Zielfunktion, Nebenbedingungen, Gradienten UND Hesse-Matrizen (Kopie aus
  `lagrange-kkt-demo`/`straf-barriere-demo`).
- `sqp_reference.py` – Referenzlösung (Kopie der Hand-Formel aus Stück 4).
- `sqp_qp.py` – löst die kleine QP-Teilaufgabe per aktive-Menge-Enumeration (inaktiv/aktiv),
  analog zu `kkt_solver.py`, hier je Iteration neu geprüft.
- `sqp_solver.py` – äußere Iteration: voller Schritt (kein Line-Search nötig), informierter
  Standard-Startwert wie Stück 4.
- `sqp_evaluation.py` – Newton-KKT-Reduktions-Check, Referenz-Konvergenz + Robustheit,
  linear-restringiertes Hilfsproblem, Iterationszahl-Vergleich (eigene, schlanke Kopien der
  Straf-/Barriere-Folgen aus Stück 5), Nebenbefund-Check, Gradienten-Check.
- `sqp_visualization.py` – Plotly: Trajektorie im $(r,h)$-Raum, Balkendiagramm
  Iterationszahl-Vergleich.

## Was die App zeigt

Volumen und Höhenlimit in der Sidebar; Kontur mit Volumen-Kurve, Höhenlimit-Linie und SQP-
Trajektorie; Fall, $r^*$, $h^*$, $\lambda$, $\mu$ und Iterationszahl als Metriken; der
Iterationszahl-Vergleich (SQP vs. Straf vs. Barriere) als zentrale Messung; ein
"📐"-Abschnitt mit der vollständigen Korrektheits-Kette.

## Was nicht funktioniert hat / Grenzen

**Zwei echte, ungeplante Funde:** (1) ein neutraler Startwert für $\lambda$ kann zu einem
vorübergehenden Fehlgriff bei der aktiven Menge führen (behoben durch den informierten
Startwert aus Stück 4). (2) SQP kann — wie SciPy in Stück 4 — zu einem mathematisch gültigen,
aber physikalisch unsinnigen KKT-Punkt mit negativem Radius konvergieren, wenn der Startpunkt
ungünstig ist; die KKT-Bedingungen allein kodieren keine Domänen-Einschränkung wie $r\ge0$.

**Grenzen:** kein Line-Search/Merit-Function-Mechanismus (bewusst weggelassen, da Vehikel B mit
informiertem Start immer mit dem vollen Schritt konvergiert) — ein industrietauglicher SQP-Löser
braucht das für größere, schlechter konditionierte Probleme. Nur eine Ungleichung (wie Stück 4/5).

## Tests

56 Tests, `python -m pytest tests/ -v` (Laufzeit lokal ~5 Sekunden):
- `test_functions.py` – Zielfunktion/Nebenbedingungen, Gradient/Hesse-Matrix gegen finite
  Differenzen.
- `test_reference.py` – Referenzlösung.
- `test_qp.py` – QP-Teilaufgabe (Fallauswahl, KKT-System exakt gelöst).
- `test_solver.py` – Konvergenz, Trajektorie, naiver Start braucht mehr Iterationen.
- `test_oracle_sqp.py` – unabhängige Orakel: Reduktion auf eine Variable $r$ (Lösung und Multiplikatoren),
  QP-Teilaufgabe über die Parametrisierung der Gleichungsgeraden, Straf-Ableitungen per finiter Differenz,
  Barriere-Stufenzahl per Goldener-Schnitt-Suche.
- `test_evaluation.py`, `test_claims.py` – jede Zahl oben nachgerechnet.
- `test_presets.py`, `test_app.py` – Presets, Regler-Extremwerte, Footer.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `sqp_constants.py` | Regler-Grenzen, Presets |
| `sqp_functions.py` | Zielfunktion, Nebenbedingungen, Hesse-Matrizen |
| `sqp_reference.py` | Referenzlösung (Kopie aus Stück 4) |
| `sqp_qp.py` | QP-Teilaufgabe (aktive-Menge-Enumeration) |
| `sqp_solver.py` | Äußere SQP-Iteration |
| `sqp_evaluation.py` | Korrektheits-Kette |
| `sqp_visualization.py` | Plotly-Plots |
| `sqp_presets.py` | Permalink-Sync, Presets |
| `tests/` | pytest-Suite |

## Bewusst nicht umgesetzt

Kein Line-Search/Trust-Region/Merit-Function-Mechanismus zur Globalisierung (bewusst — Vehikel B
ist klein genug, dass der volle Newton-Schritt mit informiertem Start immer konvergiert; ein
industrietauglicher Löser braucht das für größere Probleme). Kein allgemeiner QP-Löser für
beliebig viele Nebenbedingungen — Innere-Punkte-Verfahren (Stück 7) sind die
systematische Antwort darauf.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements-dev.txt
streamlit run app.py
```

## Literatur

- Wilson, R. B. (1963). *A Simplicial Algorithm for Concave Programming.* PhD-Thesis, Harvard
  University.
- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization* (2. Aufl.). Springer, Kap. 18.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Nichtlineare Optimierung: acht Stücke, zwei Äste](https://sebastianhanisch.net/konzepte-nichtlineare-optimierung.html).
