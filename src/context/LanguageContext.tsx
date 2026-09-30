import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

export type Language = "en" | "es" | "fr" | "de" | "hi" | "ja";

export interface LanguageInfo {
  code: Language;
  name: string;
  nativeName: string;
  flag: string;
}

export const SUPPORTED_LANGUAGES: LanguageInfo[] = [
  { code: "en", name: "English", nativeName: "English", flag: "🇺🇸" },
  { code: "es", name: "Spanish", nativeName: "Español", flag: "🇪🇸" },
  { code: "fr", name: "French", nativeName: "Français", flag: "🇫🇷" },
  { code: "de", name: "German", nativeName: "Deutsch", flag: "🇩🇪" },
  { code: "hi", name: "Hindi", nativeName: "हिन्दी", flag: "🇮🇳" },
  { code: "ja", name: "Japanese", nativeName: "日本語", flag: "🇯🇵" },
];

export const translations: Record<Language, Record<string, string>> = {
  en: {
    // Brand & Header
    "brand.title": "NETWORLD",
    "brand.subtitle": "CYBER FORECAST SYSTEM",
    "header.dataset": "CIC-IDS2018",
    "header.trafficLoaded": "TRAFFIC LOADED",
    "header.awaitingCsv": "AWAITING CSV",
    "header.reset": "Reset",
    "header.uploadTraffic": "Upload Traffic",
    "header.uploading": "Processing...",
    "header.runForecast": "Run Forecast",
    "header.forecasting": "Running...",
    "header.toggleTheme": "Toggle Theme",
    "header.selectLanguage": "Select Language",
    "header.themeDark": "Dark Mode",
    "header.themeLight": "Light Mode",

    // Navigation
    "nav.commandCenter": "Command Center",
    "nav.trafficAnalysis": "Traffic Analysis",
    "nav.worldModel": "World Model",
    "nav.attackForecast": "Attack Forecast",
    "nav.digitalTwin": "Digital Twin",
    "nav.whatIf": "What-If Simulator",
    "nav.response": "Defender Response",
    "nav.explainability": "Explainability",
    "nav.validation": "Validation",

    // Systems
    "systems.title": "SYSTEMS",
    "systems.worldModel": "World Model",
    "systems.inferenceEngine": "Inference Engine",
    "systems.shapEngine": "SHAP Engine",
    "systems.offlineMode": "Offline Mode",
    "systems.online": "ONLINE",
    "systems.active": "ACTIVE",

    // KPI Cards & Metrics
    "kpi.currentRisk": "Current Risk",
    "kpi.forecastRisk": "Forecast Risk",
    "kpi.mitreStage": "Predicted MITRE Stage",
    "kpi.worldModelEngine": "World Model Engine",
    "kpi.observedSequence": "Observed sequence state",
    "kpi.lookaheadProjection": "T+10 peak projection",
    "kpi.confidence": "confidence",
    "kpi.featuresCount": "36 Temporal Flow Features",
    "kpi.analyzed": "ANALYZED",
    "kpi.awaitingUpload": "AWAITING CSV UPLOAD",
    "kpi.noTraffic": "NO TRAFFIC",
    "kpi.standby": "STANDBY",
    "kpi.ready": "READY",

    // Common
    "common.riskHigh": "HIGH RISK",
    "common.riskMedium": "MEDIUM RISK",
    "common.riskLow": "LOW RISK",
    "common.benign": "Benign",
    "common.attack": "Attack",
    "common.infiltration": "Infiltration",
    "common.flows": "flows",
  },
  es: {
    // Brand & Header
    "brand.title": "NETWORLD",
    "brand.subtitle": "SISTEMA DE PREDICCIÓN CIBERNÉTICA",
    "header.dataset": "CIC-IDS2018",
    "header.trafficLoaded": "TRÁFICO CARGADO",
    "header.awaitingCsv": "ESPERANDO CSV",
    "header.reset": "Restablecer",
    "header.uploadTraffic": "Subir Tráfico",
    "header.uploading": "Procesando...",
    "header.runForecast": "Ejecutar Predicción",
    "header.forecasting": "Ejecutando...",
    "header.toggleTheme": "Cambiar Tema",
    "header.selectLanguage": "Seleccionar Idioma",
    "header.themeDark": "Modo Oscuro",
    "header.themeLight": "Modo Claro",

    // Navigation
    "nav.commandCenter": "Centro de Comando",
    "nav.trafficAnalysis": "Análisis de Tráfico",
    "nav.worldModel": "Modelo del Mundo",
    "nav.attackForecast": "Pronóstico de Ataques",
    "nav.digitalTwin": "Gemelo Digital",
    "nav.whatIf": "Simulador What-If",
    "nav.response": "Respuesta Defensiva",
    "nav.explainability": "Explicabilidad",
    "nav.validation": "Validación",

    // Systems
    "systems.title": "SISTEMAS",
    "systems.worldModel": "Modelo del Mundo",
    "systems.inferenceEngine": "Motor de Inferencia",
    "systems.shapEngine": "Motor SHAP",
    "systems.offlineMode": "Modo Desconectado",
    "systems.online": "EN LÍNEA",
    "systems.active": "ACTIVO",

    // KPI Cards & Metrics
    "kpi.currentRisk": "Riesgo Actual",
    "kpi.forecastRisk": "Riesgo Proyectado",
    "kpi.mitreStage": "Fase MITRE Estimada",
    "kpi.worldModelEngine": "Motor del Modelo",
    "kpi.observedSequence": "Estado de secuencia observada",
    "kpi.lookaheadProjection": "Proyección máxima T+10",
    "kpi.confidence": "confianza",
    "kpi.featuresCount": "36 Características Temporales",
    "kpi.analyzed": "ANALIZADO",
    "kpi.awaitingUpload": "ESPERANDO CARGA DE CSV",
    "kpi.noTraffic": "SIN TRÁFICO",
    "kpi.standby": "EN ESPERA",
    "kpi.ready": "LISTO",

    // Common
    "common.riskHigh": "RIESGO ALTO",
    "common.riskMedium": "RIESGO MEDIO",
    "common.riskLow": "RIESGO BAJO",
    "common.benign": "Benigno",
    "common.attack": "Ataque",
    "common.infiltration": "Infiltración",
    "common.flows": "flujos",
  },
  fr: {
    // Brand & Header
    "brand.title": "NETWORLD",
    "brand.subtitle": "SYSTÈME DE PRÉVISION CYBER",
    "header.dataset": "CIC-IDS2018",
    "header.trafficLoaded": "TRAFIC CHARGÉ",
    "header.awaitingCsv": "EN ATTENTE CSV",
    "header.reset": "Réinitialiser",
    "header.uploadTraffic": "Téléverser Trafic",
    "header.uploading": "Traitement...",
    "header.runForecast": "Lancer Prévision",
    "header.forecasting": "Calcul...",
    "header.toggleTheme": "Changer Thème",
    "header.selectLanguage": "Choisir Langue",
    "header.themeDark": "Mode Sombre",
    "header.themeLight": "Mode Clair",

    // Navigation
    "nav.commandCenter": "Centre de Commandement",
    "nav.trafficAnalysis": "Analyse du Trafic",
    "nav.worldModel": "Modèle du Monde",
    "nav.attackForecast": "Prévision des Attaques",
    "nav.digitalTwin": "Jumeau Numérique",
    "nav.whatIf": "Simulateur What-If",
    "nav.response": "Réponse Défensive",
    "nav.explainability": "Explicabilité",
    "nav.validation": "Validation",

    // Systems
    "systems.title": "SYSTÈMES",
    "systems.worldModel": "Modèle du Monde",
    "systems.inferenceEngine": "Moteur d'Inférence",
    "systems.shapEngine": "Moteur SHAP",
    "systems.offlineMode": "Mode Hors Ligne",
    "systems.online": "EN LIGNE",
    "systems.active": "ACTIF",

    // KPI Cards & Metrics
    "kpi.currentRisk": "Risque Actuel",
    "kpi.forecastRisk": "Risque Prévu",
    "kpi.mitreStage": "Phase MITRE Estimée",
    "kpi.worldModelEngine": "Moteur PyTorch LSTM",
    "kpi.observedSequence": "État de la séquence observée",
    "kpi.lookaheadProjection": "Projection pic T+10",
    "kpi.confidence": "confiance",
    "kpi.featuresCount": "36 Caractéristiques Temporelles",
    "kpi.analyzed": "ANALYSÉ",
    "kpi.awaitingUpload": "EN ATTENTE DE FICHIER CSV",
    "kpi.noTraffic": "AUCUN TRAFIC",
    "kpi.standby": "EN VEILLE",
    "kpi.ready": "PRÊT",

    // Common
    "common.riskHigh": "RISQUE ÉLEVÉ",
    "common.riskMedium": "RISQUE MOYEN",
    "common.riskLow": "RISQUE FAIBLE",
    "common.benign": "Bénin",
    "common.attack": "Attaque",
    "common.infiltration": "Infiltration",
    "common.flows": "flux",
  },
  de: {
    // Brand & Header
    "brand.title": "NETWORLD",
    "brand.subtitle": "CYBER-PROGNOSE-SYSTEM",
    "header.dataset": "CIC-IDS2018",
    "header.trafficLoaded": "VERKEHR GELADEN",
    "header.awaitingCsv": "CSV ERWARTET",
    "header.reset": "Zurücksetzen",
    "header.uploadTraffic": "Verkehr Hochladen",
    "header.uploading": "Verarbeitung...",
    "header.runForecast": "Prognose Starten",
    "header.forecasting": "Berechnung...",
    "header.toggleTheme": "Design Wechseln",
    "header.selectLanguage": "Sprache Wählen",
    "header.themeDark": "Dunkelmodus",
    "header.themeLight": "Hellmodus",

    // Navigation
    "nav.commandCenter": "Kommandozentrale",
    "nav.trafficAnalysis": "Verkehrsanalyse",
    "nav.worldModel": "Weltmodell",
    "nav.attackForecast": "Angriffsprognose",
    "nav.digitalTwin": "Digitaler Zwilling",
    "nav.whatIf": "What-If-Simulator",
    "nav.response": "Abwehrreaktion",
    "nav.explainability": "Erklärbarkeit",
    "nav.validation": "Validierung",

    // Systems
    "systems.title": "SYSTEME",
    "systems.worldModel": "Weltmodell",
    "systems.inferenceEngine": "Inferenz-Engine",
    "systems.shapEngine": "SHAP-Engine",
    "systems.offlineMode": "Offline-Modus",
    "systems.online": "ONLINE",
    "systems.active": "AKTIV",

    // KPI Cards & Metrics
    "kpi.currentRisk": "Aktuelles Risiko",
    "kpi.forecastRisk": "Prognostiziertes Risiko",
    "kpi.mitreStage": "Geschätzte MITRE-Phase",
    "kpi.worldModelEngine": "PyTorch LSTM Engine",
    "kpi.observedSequence": "Beobachteter Sequenzstatus",
    "kpi.lookaheadProjection": "T+10 Spitzenprojektion",
    "kpi.confidence": "Zuversicht",
    "kpi.featuresCount": "36 Zeitliche Flussmerkmale",
    "kpi.analyzed": "ANALYSIERT",
    "kpi.awaitingUpload": "WARTET AUF CSV-UPLOAD",
    "kpi.noTraffic": "KEIN VERKEHR",
    "kpi.standby": "STANDBY",
    "kpi.ready": "BEREIT",

    // Common
    "common.riskHigh": "HOHES RISIKO",
    "common.riskMedium": "MITTLERES RISIKO",
    "common.riskLow": "GERINGES RISIKO",
    "common.benign": "Gutartig",
    "common.attack": "Angriff",
    "common.infiltration": "Infiltration",
    "common.flows": "Flüsse",
  },
  hi: {
    // Brand & Header
    "brand.title": "NETWORLD",
    "brand.subtitle": "साइबर पूर्वानुमान प्रणाली",
    "header.dataset": "CIC-IDS2018",
    "header.trafficLoaded": "ट्रैफ़िक लोड हुआ",
    "header.awaitingCsv": "CSV की प्रतीक्षा",
    "header.reset": "रीसेट",
    "header.uploadTraffic": "ट्रैफ़िक अपलोड करें",
    "header.uploading": "प्रक्रिया जारी...",
    "header.runForecast": "पूर्वानुमान चलाएं",
    "header.forecasting": "चल रहा है...",
    "header.toggleTheme": "थीम बदलें",
    "header.selectLanguage": "भाषा चुनें",
    "header.themeDark": "डार्क मोड",
    "header.themeLight": "लाइट मोड",

    // Navigation
    "nav.commandCenter": "कमांड सेंटर",
    "nav.trafficAnalysis": "ट्रैफ़िक विश्लेषण",
    "nav.worldModel": "वर्ल्ड मॉडल",
    "nav.attackForecast": "हमला पूर्वानुमान",
    "nav.digitalTwin": "डिजिटल ट्विन",
    "nav.whatIf": "व्हाट-इफ़ सिम्युलेटर",
    "nav.response": "डिफेंडर प्रतिक्रिया",
    "nav.explainability": "स्पष्टीकरण (Explainability)",
    "nav.validation": "सत्यापन (Validation)",

    // Systems
    "systems.title": "प्रणालियाँ (Systems)",
    "systems.worldModel": "वर्ल्ड मॉडल",
    "systems.inferenceEngine": "इनफेरेंस इंजन",
    "systems.shapEngine": "SHAP इंजन",
    "systems.offlineMode": "ऑफ़लाइन मोड",
    "systems.online": "ऑनलाइन",
    "systems.active": "सक्रिय",

    // KPI Cards & Metrics
    "kpi.currentRisk": "वर्तमान जोखिम",
    "kpi.forecastRisk": "अनुमानित जोखिम",
    "kpi.mitreStage": "अनुमानित MITRE चरण",
    "kpi.worldModelEngine": "वर्ल्ड मॉडल इंजन",
    "kpi.observedSequence": "अवलोकित अनुक्रम स्थिति",
    "kpi.lookaheadProjection": "T+10 पीक प्रक्षेपण",
    "kpi.confidence": "विश्वास",
    "kpi.featuresCount": "36 अस्थायी प्रवाह विशेषताएँ",
    "kpi.analyzed": "विश्लेषित",
    "kpi.awaitingUpload": "CSV अपलोड की प्रतीक्षा है",
    "kpi.noTraffic": "कोई ट्रैफ़िक नहीं",
    "kpi.standby": "स्टैंडबाय",
    "kpi.ready": "तैयार",

    // Common
    "common.riskHigh": "उच्च जोखिम",
    "common.riskMedium": "मध्यम जोखिम",
    "common.riskLow": "कम जोखिम",
    "common.benign": "सामान्य (Benign)",
    "common.attack": "हमला (Attack)",
    "common.infiltration": "घुसपैठ (Infiltration)",
    "common.flows": "प्रवाह",
  },
  ja: {
    // Brand & Header
    "brand.title": "NETWORLD",
    "brand.subtitle": "サイバー予測システム",
    "header.dataset": "CIC-IDS2018",
    "header.trafficLoaded": "トラフィック読込済",
    "header.awaitingCsv": "CSV待機中",
    "header.reset": "リセット",
    "header.uploadTraffic": "トラフィックをアップロード",
    "header.uploading": "処理中...",
    "header.runForecast": "予測を実行",
    "header.forecasting": "実行中...",
    "header.toggleTheme": "テーマ切替",
    "header.selectLanguage": "言語選択",
    "header.themeDark": "ダークモード",
    "header.themeLight": "ライトモード",

    // Navigation
    "nav.commandCenter": "コマンドセンター",
    "nav.trafficAnalysis": "トラフィック分析",
    "nav.worldModel": "ワールドモデル",
    "nav.attackForecast": "攻撃予測",
    "nav.digitalTwin": "デジタルツイン",
    "nav.whatIf": "What-If シミュレーター",
    "nav.response": "防衛レスポンス",
    "nav.explainability": "説明可能性 (Explainability)",
    "nav.validation": "モデル検証",

    // Systems
    "systems.title": "稼働システム",
    "systems.worldModel": "ワールドモデル",
    "systems.inferenceEngine": "推論エンジン",
    "systems.shapEngine": "SHAP エンジン",
    "systems.offlineMode": "オフラインモード",
    "systems.online": "オンライン",
    "systems.active": "アクティブ",

    // KPI Cards & Metrics
    "kpi.currentRisk": "現在のリスク",
    "kpi.forecastRisk": "予測リスク",
    "kpi.mitreStage": "推定 MITRE ステージ",
    "kpi.worldModelEngine": "ワールドモデル・エンジン",
    "kpi.observedSequence": "観測シーケンス状態",
    "kpi.lookaheadProjection": "T+10 ピーク予測",
    "kpi.confidence": "信頼度",
    "kpi.featuresCount": "36時系列フロー特徴量",
    "kpi.analyzed": "分析完了",
    "kpi.awaitingUpload": "CSVアップロード待機中",
    "kpi.noTraffic": "トラフィックなし",
    "kpi.standby": "待機中",
    "kpi.ready": "準備完了",

    // Common
    "common.riskHigh": "高リスク",
    "common.riskMedium": "中リスク",
    "common.riskLow": "低リスク",
    "common.benign": "正常 (Benign)",
    "common.attack": "攻撃 (Attack)",
    "common.infiltration": "侵入 (Infiltration)",
    "common.flows": "フロー",
  },
};

interface LanguageContextType {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: (key: string, defaultVal?: string) => string;
  languages: LanguageInfo[];
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [language, setLanguageState] = useState<Language>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("networld_lang") as Language | null;
      if (saved && translations[saved]) {
        return saved;
      }
      const navLang = navigator.language.slice(0, 2) as Language;
      if (translations[navLang]) {
        return navLang;
      }
    }
    return "en";
  });

  useEffect(() => {
    localStorage.setItem("networld_lang", language);
    document.documentElement.lang = language;
  }, [language]);

  const setLanguage = (lang: Language) => {
    if (translations[lang]) {
      setLanguageState(lang);
    }
  };

  const t = (key: string, defaultVal?: string): string => {
    const langDict = translations[language];
    if (langDict && langDict[key]) {
      return langDict[key];
    }
    const fallbackDict = translations.en;
    if (fallbackDict && fallbackDict[key]) {
      return fallbackDict[key];
    }
    return defaultVal || key;
  };

  return (
    <LanguageContext.Provider
      value={{
        language,
        setLanguage,
        t,
        languages: SUPPORTED_LANGUAGES,
      }}
    >
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage(): LanguageContextType {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error("useLanguage must be used within a LanguageProvider");
  }
  return context;
}
