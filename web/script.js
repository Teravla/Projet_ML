const API_BASE = 'http://localhost:5000/api';

// État de la pagination
let decisionsState = {
    offset: 0,
    limit: 10,
    total: 0,
    allDecisions: []
};

// Vérifier la connexion à l'API au chargement
window.addEventListener('load', async () => {
    await checkApiHealth();
    await checkModelStatus();
    await loadAllData();
});

async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`, {
            headers: { 'Accept': 'application/json' }
        });
        if (response.ok) {
            document.getElementById('api-status').textContent = 'API Connectée';
            document.getElementById('status-indicator').classList.remove('status-offline');
        } else {
            setApiError();
        }
    } catch (error) {
        setApiError();
    }
}

function setApiError() {
    document.getElementById('api-status').textContent = 'API Indisponible - Vérifiez que le serveur Flask est lancé';
    document.getElementById('status-indicator').classList.add('status-offline');
}

async function checkModelStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const data = await response.json();

        if (!data.model_loaded) {
            // Afficher l'alerte si pas de modèle
            document.getElementById('model-alert').style.display = 'block';
            document.getElementById('api-status').innerHTML += ' <span style="color: orange;">(Données simulées - Aucun modèle)</span>';
        } else {
            // Masquer l'alerte si le modèle est chargé
            document.getElementById('model-alert').style.display = 'none';
            document.getElementById('api-status').innerHTML += ' <span style="color: green;">(Modèle: ' + data.model_filename + ')</span>';
        }
    } catch (error) {
        console.error('Erreur lors de la vérification du statut du modèle:', error);
    }
}

// === GESTION PAGE D'ENTRAÎNEMENT ===

// Configuration d'entraînement
let trainingConfig = {
    imageSize: 64,
    algos: { cnn: true, transfer: true, mlp: false },
    epochs: 100,
    useSweep: false,
    sweepTrials: 4,
    sweepEpochs: 12,
    useFinalTrain: false,
    finalEpochs: 40,
    useTTA: true,
    useFocal: false,
};

// Estimations de temps (en minutes)
const TIME_ESTIMATES = {
    cnn_base: { 64: 1, 128: 2, 224: 5 },  // par epoch
    cnn_sweep: { 64: 2, 128: 4, 224: 8 }, // par trial
    transfer: { 64: 1.5, 128: 3, 224: 6 }, // par epoch
    mlp: { 64: 0.5, 128: 0.8, 224: 1.2 }, // par epoch
};

function showTrainingPage() {
    document.getElementById('training-page').style.display = 'block';
    document.getElementById('model-alert').style.display = 'none';
    updateCommand();
    window.scrollTo(0, 0);
}

function hideTrainingPage() {
    document.getElementById('training-page').style.display = 'none';
    document.getElementById('model-alert').style.display = 'block';
}

function setImageSize(size) {
    trainingConfig.imageSize = size;

    // Mettre à jour les boutons
    document.querySelectorAll('.img-size-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');

    updateCommand();
}

function toggleSweepOptions() {
    const useSweep = document.getElementById('use-sweep').checked;
    trainingConfig.useSweep = useSweep;
    document.getElementById('sweep-options').style.display = useSweep ? 'block' : 'none';
    document.getElementById('epochs-standard').style.display = useSweep ? 'none' : 'block';
    updateCommand();
}

function toggleFinalTrainOptions() {
    const useFinalTrain = document.getElementById('use-final-train').checked;
    trainingConfig.useFinalTrain = useFinalTrain;
    document.getElementById('final-train-options').style.display = useFinalTrain ? 'block' : 'none';
    updateCommand();
}

function updateCommand() {
    // Récupérer les valeurs actuelles
    trainingConfig.algos.cnn = document.getElementById('algo-cnn').checked;
    trainingConfig.algos.transfer = document.getElementById('algo-transfer').checked;
    trainingConfig.algos.mlp = document.getElementById('algo-mlp').checked;
    trainingConfig.epochs = parseInt(document.getElementById('epochs-slider').value);
    trainingConfig.sweepTrials = parseInt(document.getElementById('sweep-trials').value);
    trainingConfig.sweepEpochs = parseInt(document.getElementById('sweep-epochs').value);
    trainingConfig.finalEpochs = parseInt(document.getElementById('final-epochs').value);
    trainingConfig.useTTA = document.getElementById('use-tta').checked;
    trainingConfig.useFocal = document.getElementById('use-focal').checked;

    // Générer la commande
    let cmd = 'poetry run task base64';

    // Ajouter les algos
    if (trainingConfig.algos.cnn) cmd += ' --cnn';
    if (trainingConfig.algos.mlp) cmd += ' --mlp';
    if (trainingConfig.algos.transfer) cmd += ' --transfer';

    // Ajouter les options
    cmd += ` --img-size ${trainingConfig.imageSize}`;

    if (trainingConfig.useSweep) {
        cmd += ` --sweep --sweep-trials ${trainingConfig.sweepTrials} --sweep-epochs ${trainingConfig.sweepEpochs}`;
    } else {
        cmd += ` --epochs ${trainingConfig.epochs}`;
    }

    if (trainingConfig.useFinalTrain) {
        cmd += ` --final-train --final-epochs ${trainingConfig.finalEpochs}`;
    }

    if (trainingConfig.useTTA) cmd += ' --tta';
    if (trainingConfig.useFocal) cmd += ' --focal';

    document.getElementById('command-display').textContent = cmd;

    // Mettre à jour le résumé
    updateConfigSummary();

    // Mettre à jour l'estimation de temps
    updateTimeEstimate();
}

function updateConfigSummary() {
    const algos = [];
    if (trainingConfig.algos.cnn) algos.push('CNN');
    if (trainingConfig.algos.transfer) algos.push('Transfer');
    if (trainingConfig.algos.mlp) algos.push('MLP');

    const options = [];
    if (trainingConfig.useSweep) options.push(`Sweep (${trainingConfig.sweepTrials}×${trainingConfig.sweepEpochs}ep)`);
    if (trainingConfig.useFinalTrain) options.push(`Final-train (${trainingConfig.finalEpochs}ep)`);
    if (trainingConfig.useTTA) options.push('TTA');
    if (trainingConfig.useFocal) options.push('Focal Loss');
    if (!trainingConfig.useSweep) options.push(`${trainingConfig.epochs} epochs`);

    let summary = `
        <div>Image: <strong>${trainingConfig.imageSize}x${trainingConfig.imageSize}</strong></div>
        <div>Algos: <strong>${algos.join(' + ')}</strong></div>
        <div>Options: <strong>${options.length > 0 ? options.join(', ') : 'Aucune'}</strong></div>
    `;

    document.getElementById('config-summary').innerHTML = summary;
}

function updateTimeEstimate() {
    let totalTime = 0;
    let breakdown = [];

    // Estimer le temps pour CNN
    if (trainingConfig.algos.cnn) {
        let cnnTime;
        if (trainingConfig.useSweep) {
            cnnTime = trainingConfig.sweepTrials * trainingConfig.sweepEpochs * TIME_ESTIMATES.cnn_sweep[trainingConfig.imageSize];
            breakdown.push(`CNN sweep: ${cnnTime.toFixed(0)}min`);
        } else {
            cnnTime = trainingConfig.epochs * TIME_ESTIMATES.cnn_base[trainingConfig.imageSize];
            breakdown.push(`CNN: ${cnnTime.toFixed(0)}min`);
        }
        if (trainingConfig.useFinalTrain) {
            const finalTime = trainingConfig.finalEpochs * TIME_ESTIMATES.cnn_base[trainingConfig.imageSize];
            cnnTime += finalTime;
            breakdown.push(`Final-train: ${finalTime.toFixed(0)}min`);
        }
        totalTime += cnnTime;
    }

    // Transfer Learning
    if (trainingConfig.algos.transfer) {
        let transferTime = trainingConfig.epochs * TIME_ESTIMATES.transfer[trainingConfig.imageSize];
        breakdown.push(`Transfer: ${transferTime.toFixed(0)}min`);
        totalTime += transferTime;
    }

    // MLP
    if (trainingConfig.algos.mlp) {
        let mlpTime = trainingConfig.epochs * TIME_ESTIMATES.mlp[trainingConfig.imageSize];
        breakdown.push(`MLP: ${mlpTime.toFixed(0)}min`);
        totalTime += mlpTime;
    }

    // Ajouter du temps si TTA (post-training)
    if (trainingConfig.useTTA) {
        totalTime *= 1.3;
        breakdown.push(`TTA: +30% temps`);
    }

    // Formater l'estimation
    let estimate;
    if (totalTime < 60) {
        estimate = `~${Math.round(totalTime)} minutes`;
    } else if (totalTime < 1440) {
        const hours = (totalTime / 60).toFixed(1);
        estimate = `~${hours} heures`;
    } else {
        const days = (totalTime / 1440).toFixed(1);
        estimate = `~${days} jours`;
    }

    document.getElementById('time-estimate').textContent = estimate;
    document.getElementById('time-breakdown').textContent = breakdown.join(' + ');
}

function copyCommand() {
    const cmd = document.getElementById('command-display').textContent;
    navigator.clipboard.writeText(cmd).then(() => {
        alert('Commande copiée dans le presse-papiers!');
    }).catch(err => {
        console.error('Erreur:', err);
        alert('Erreur lors de la copie');
    });
}

function executeTraining() {
    const cmd = document.getElementById('command-display').textContent;
    alert(
        'INSTRUCTIONS:\n\n' +
        '1. Ouvrez un terminal PowerShell\n' +
        '2. Naviguez jusqu\'au répertoire du projet:\n' +
        '   cd c:\\Users\\valen\\Documents\\EFREI\\I2\\Machine_Learning\\Projet\n\n' +
        '3. Exécutez la commande:\n' +
        '   ' + cmd + '\n\n' +
        '4. Patientez l\'entraînement (peut prendre plusieurs heures)\n' +
        '5. Une fois terminé, rafraîchissez le dashboard pour charger le modèle\n\n' +
        'La commande a été copiée. Appuyez sur Ctrl+C pour fermer ce message et coller dans le terminal.'
    );
    copyCommand();
}

async function loadAllData() {
    await loadStats();
    await loadDecisions();
    await loadMetrics();
}

async function loadStats() {
    const container = document.getElementById('stats-content');
    try {
        const response = await fetch(`${API_BASE}/stats`);
        const data = await response.json();

        let html = `
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-label">Total Patients</div>
                <div class="metric-value">${data.n_total}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Couverture Automatique</div>
                <div class="metric-value">${data.couverture_automatique}%</div>
                <div class="metric-subtext">${data.n_total - data.revisions} cas auto</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Alertes Sécurité</div>
                <div class="metric-value" style="color: #e74c3c;">${data.alertes}</div>
                <div class="metric-subtext">Révisions notumor</div>
            </div>
        </div>

        <h3>Distribution par Confiance</h3>
        <div class="distribution-grid">
    `;

        for (const [level, info] of Object.entries(data.confiance_distribution)) {
            const badge = `badge-${level.toLowerCase()}`;
            html += `
            <div class="dist-item">
                <span class="badge ${badge}">${level}</span>
                <div class="dist-count">${info.count}</div>
                <div class="dist-percent">${info.percent}%</div>
            </div>
        `;
        }

        html += `</div><h3>Distribution par Priorité Triage</h3><div class="distribution-grid">`;

        for (const [level, info] of Object.entries(data.priorite_distribution)) {
            const badgeMap = {
                'URGENTE': 'urgente',
                'ELEVEE': 'elevee',
                'NORMALE': 'normale',
                'ROUTINE': 'routine'
            };
            const badge = `badge-${badgeMap[level] || 'routine'}`;
            html += `
            <div class="dist-item">
                <span class="badge ${badge}">${level}</span>
                <div class="dist-count">${info.count}</div>
                <div class="dist-percent">${info.percent}%</div>
            </div>
        `;
        }

        html += `</div>`;
        container.innerHTML = html;
    } catch (error) {
        container.innerHTML = `<div class="error-message">Erreur lors du chargement des statistiques: ${error.message}</div>`;
    }
}

async function loadDecisions() {
    // Réinitialiser l'état pour un premier chargement
    decisionsState.offset = 0;
    decisionsState.allDecisions = [];
    await loadMoreDecisions();
}

async function loadMoreDecisions() {
    const container = document.getElementById('decisions-content');
    try {
        const response = await fetch(`${API_BASE}/decisions?limit=${decisionsState.limit}&offset=${decisionsState.offset}`);
        const data = await response.json();

        decisionsState.total = data.total;
        decisionsState.allDecisions = decisionsState.allDecisions.concat(data.decisions);
        decisionsState.offset += data.decisions.length;

        renderDecisionsTable();
    } catch (error) {
        container.innerHTML = `<div class="error-message">Erreur: ${error.message}</div>`;
    }
}

function renderDecisionsTable() {
    const container = document.getElementById('decisions-content');

    let html = `
        <div class="alert-box">
            Affichage: <strong>${decisionsState.allDecisions.length}</strong> décisions sur <strong>${decisionsState.total}</strong> total
        </div>
        <div class="table-responsive">
            <table>
                <thead>
                    <tr>
                        <th>Patient</th>
                        <th>Diagnostic</th>
                        <th>Confiance</th>
                        <th>Niveau</th>
                        <th>Priorité</th>
                        <th>Alerte</th>
                        <th>Décision</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
        `;

    for (const d of decisionsState.allDecisions) {
        const badgeLevel = `badge-${d.niveau_confiance.toLowerCase()}`;
        const badgePrio = d.priorite.includes('URGENTE') ? 'urgente' :
            d.priorite.includes('Élevée') ? 'elevee' :
                d.priorite.includes('Normale') ? 'normale' : 'routine';
        const alerte = d.alerte_securite ? 'OUI' : 'Non';

        html += `
            <tr>
                <td><strong>${d.patient_id}</strong></td>
                <td>${d.classe_predite}</td>
                <td>${(d.confiance * 100).toFixed(1)}%</td>
                <td><span class="badge ${badgeLevel}">${d.niveau_confiance}</span></td>
                <td><span class="badge badge-${badgePrio}">${d.priorite}</span></td>
                <td>${alerte}</td>
                <td style="font-size: 11px;">${d.decision}</td>
                <td>
                    <button class="btn-pdf" onclick="generatePDF('${d.patient_id}')" title="Générer rapport PDF">
                        📄 PDF
                    </button>
                </td>
            </tr>
        `;
    }

    html += `
                </tbody>
            </table>
        </div>
    `;

    // Ajouter le bouton Load More si nécessaire
    if (decisionsState.allDecisions.length < decisionsState.total) {
        const remaining = decisionsState.total - decisionsState.allDecisions.length;
        html += `
            <div style="text-align: center; margin-top: 20px;">
                <button class="btn" onclick="loadMoreDecisions()" style="padding: 12px 30px; font-size: 14px;">
                    Charger plus (${remaining} restants)
                </button>
            </div>
        `;
    }

    container.innerHTML = html;
}

function generatePDF(patientId) {
    // Ouvrir le PDF dans une nouvelle fenêtre
    window.open(`${API_BASE}/rapport/${patientId}/pdf`, '_blank');
}

async function loadMetrics() {
    const container = document.getElementById('metrics-content');
    try {
        const response = await fetch(`${API_BASE}/metrics`);
        const metrics = await response.json();

        const objAtteint = metrics.objectif_haute_confiance.objectif_atteint ?
            '<span style="color: #28a745;">OUI</span>' :
            '<span style="color: #e74c3c;">NON</span>';

        let html = `
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-label">Accuracy Globale</div>
                <div class="metric-value">${(metrics.accuracy_globale * 100).toFixed(2)}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Couverture Automatique</div>
                <div class="metric-value">${(metrics.taux_couverture_automatique * 100).toFixed(1)}%</div>
            </div>
            <div class="metric-box" id="objective-box" style="cursor: pointer;">
                <div class="metric-label">Objectif Haute Confiance</div>
                <div class="metric-value" style="font-size: 18px;">${objAtteint}</div>
                <div class="metric-subtext">N=${metrics.objectif_haute_confiance.n_cas}, Accuracy=${metrics.objectif_haute_confiance.accuracy ? (metrics.objectif_haute_confiance.accuracy * 100).toFixed(2) + '%' : 'N/A'}</div>
                <div style="font-size: 12px; color: #0984e3; font-style: italic; margin-top: 8px; font-weight: 600;">Cliquer pour en savoir plus</div>
            </div>
        </div>

        <div id="objective-modal" class="objective-modal" style="display: none;">
            <div class="objective-modal-overlay"></div>
            <div class="objective-modal-content">
                <button id="close-objective" style="position: absolute; top: 15px; right: 15px; background: none; border: none; font-size: 24px; cursor: pointer; color: #999; transition: color 0.3s;">&times;</button>
                <h4 style="color: #0984e3; margin-top: 0; font-size: 18px; margin-bottom: 15px;">Objectif Haute Confiance</h4>
                <div style="color: #333; font-size: 15px; line-height: 1.7;">
                    <p style="margin: 0 0 12px 0;">
                        <strong>Définition :</strong> Atteindre une précision &gt; 95% sur les cas où le modèle est confiant (confiance &gt; 85%)
                    </p>
                    <p style="margin: 0 0 12px 0;">
                        <strong>Critères :</strong>
                        <br/>• Seuil de confiance : <span style="background: #fff3cd; padding: 3px 8px; border-radius: 3px;"><strong>85%</strong></span>
                        <br/>• Objectif d'accuracy : <span style="background: #fff3cd; padding: 3px 8px; border-radius: 3px;"><strong>&gt; 95%</strong></span>
                    </p>
                    <p style="margin: 0 0 12px 0;">
                        <strong>Logique :</strong> Seuls les cas avec confiance &gt; 85% sont comptabilisés. Sur ces cas, le modèle doit se tromper <strong>moins de 5% du temps</strong>.
                    </p>
                    <p style="margin: 0;">
                        <strong style="color: #e74c3c;">⚠️ Importance clinique :</strong> Permet d'automatiser les décisions dangereuses en minimisant les faux négatifs graves (tumeurs non détectées). Les cas moins confiants sont revus manuellement.
                    </p>
                </div>
                <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #e0e0e0; font-size: 14px; color: #555;">
                    <strong>Résultat actuel :</strong> ${metrics.objectif_haute_confiance.n_cas} cas en haute confiance | Accuracy: ${metrics.objectif_haute_confiance.accuracy ? (metrics.objectif_haute_confiance.accuracy * 100).toFixed(2) + '%' : 'N/A'} | Objectif: <span style="color: ${metrics.objectif_haute_confiance.objectif_atteint ? '#28a745' : '#e74c3c'}; font-weight: bold;">${metrics.objectif_haute_confiance.objectif_atteint ? '✓ ATTEINT' : '✗ NON ATTEINT'}</span>
                </div>
            </div>
        </div>

        <h3>Analyse des Coûts (Phase 7)</h3>
        <div class="metrics-grid">
            <div class="metric-box" style="border-left-color: #e74c3c;">
                <div class="metric-label">Faux Négatifs (FN)</div>
                <div class="metric-value">${metrics.couts.FN}</div>
                <div class="metric-subtext">Coût: ${metrics.couts.FN} × 1000€ = ${(metrics.couts.FN * 1000).toLocaleString()}€</div>
            </div>
            <div class="metric-box" style="border-left-color: #fdcb6e;">
                <div class="metric-label">Faux Positifs (FP)</div>
                <div class="metric-value">${metrics.couts.FP}</div>
                <div class="metric-subtext">Coût: ${metrics.couts.FP} × 100€ = ${(metrics.couts.FP * 100).toLocaleString()}€</div>
            </div>
            <div class="metric-box" style="border-left-color: #0984e3;">
                <div class="metric-label">Révisions Humaines</div>
                <div class="metric-value">${metrics.couts.Revision}</div>
                <div class="metric-subtext">Coût: ${metrics.couts.Revision} × 50€ = ${(metrics.couts.Revision * 50).toLocaleString()}€</div>
            </div>
            <div class="metric-box" style="border-left-color: #667eea; background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);">
                <div class="metric-label">Coût Total SAD</div>
                <div class="metric-value">${(metrics.couts.Cost_total).toLocaleString()}€</div>
                <div class="metric-subtext">Pour ensemble patients</div>
            </div>
        </div>

        <div class="economic-analysis">
            <h3 style="color: #b71c1c; margin-bottom: 20px; font-size: 16px;">Analyse Économique du SAD</h3>
            
            <div class="formula-box">
                <div class="formula-label">Formule de Coût:</div>
                <div class="formula-content">
                    Cost = (FN × <span class="cost-weight">1000€</span>) + (FP × <span class="cost-weight">100€</span>) + (Revision × <span class="cost-weight">50€</span>)
                </div>
            </div>

            <div class="efficiency-grid">
                <div class="efficiency-item">
                    <div class="efficiency-icon">✓</div>
                    <div class="efficiency-value">${(metrics.taux_couverture_automatique * 100).toFixed(1)}%</div>
                    <div class="efficiency-label">Cas traités automatiquement</div>
                </div>
                <div class="efficiency-item">
                    <div class="efficiency-icon">▲</div>
                    <div class="efficiency-value">${metrics.objectif_haute_confiance.accuracy ? (metrics.objectif_haute_confiance.accuracy * 100).toFixed(2) : '?'}%</div>
                    <div class="efficiency-label">Précision (haute confiance)</div>
                </div>
                <div class="efficiency-item">
                    <div class="efficiency-icon">👤</div>
                    <div class="efficiency-value">${((1 - metrics.taux_couverture_automatique) * 100).toFixed(1)}%</div>
                    <div class="efficiency-label">Cas nécessitant révision</div>
                </div>
            </div>

            <div class="insight-box">
                <div class="insight-title">Impact Clinique</div>
                <div class="insight-text">
                    Le SAD automatise ${(metrics.taux_couverture_automatique * 100).toFixed(1)}% de la charge de travail tout en maintenant ${metrics.objectif_haute_confiance.accuracy ? (metrics.objectif_haute_confiance.accuracy * 100).toFixed(2) : '?'}% d'accuracy pour les cas critiques. Les ${((1 - metrics.taux_couverture_automatique) * 100).toFixed(1)}% restants bénéficient d'une révision humaine pour minimiser les faux négatifs graves et assurer la sécurité patient.
                </div>
            </div>
        </div>
    `;
        container.innerHTML = html;

        // Ajouter les event listeners pour le modal
        const objectiveBox = document.getElementById('objective-box');
        const objectiveModal = document.getElementById('objective-modal');
        const closeButton = document.getElementById('close-objective');

        if (objectiveBox && objectiveModal && closeButton) {
            // Click sur la div pour ouvrir le modal
            objectiveBox.addEventListener('click', () => {
                objectiveModal.style.display = 'flex';
            });

            // Click sur le bouton X pour fermer
            closeButton.addEventListener('click', (e) => {
                e.stopPropagation();
                objectiveModal.style.display = 'none';
            });

            // Click sur l'overlay pour fermer
            const overlay = objectiveModal.querySelector('.objective-modal-overlay');
            if (overlay) {
                overlay.addEventListener('click', () => {
                    objectiveModal.style.display = 'none';
                });
            }

            // Empêcher la fermeture quand on clique sur le contenu
            const modalContent = objectiveModal.querySelector('.objective-modal-content');
            if (modalContent) {
                modalContent.addEventListener('click', (e) => {
                    e.stopPropagation();
                });
            }

            // Fermer avec la touche Escape
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape' && objectiveModal.style.display === 'flex') {
                    objectiveModal.style.display = 'none';
                }
            });
        }
    } catch (error) {
        container.innerHTML = `<div class="error-message">Erreur: ${error.message}</div>`;
    }
}

function switchTab(event, tabName) {
    // Masquer tous les onglets
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));

    // Retirer active des boutons
    document.querySelectorAll('.nav-tab').forEach(el => el.classList.remove('active'));

    // Afficher l'onglet sélectionné
    document.getElementById(tabName).classList.add('active');
    event.target.classList.add('active');
}

async function exportData() {
    try {
        const statsRes = await fetch(`${API_BASE}/stats`);
        const decisionsRes = await fetch(`${API_BASE}/decisions`);
        const metricsRes = await fetch(`${API_BASE}/metrics`);

        const stats = await statsRes.json();
        const decisions = await decisionsRes.json();
        const metrics = await metricsRes.json();

        const exportData = {
            timestamp: new Date().toISOString(),
            stats,
            decisions,
            metrics
        };

        const json = JSON.stringify(exportData, null, 2);
        const blob = new Blob([json], {
            type: 'application/json'
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `sad-export-${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        URL.revokeObjectURL(url);
    } catch (error) {
        alert('Erreur lors de l\'export: ' + error.message);
    }
}
