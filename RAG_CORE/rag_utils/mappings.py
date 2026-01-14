from RAG_CORE.rag_utils.search_utils import norm_txt

# Mappeo de sinonimos y Servicios que se ofrece el centro, Simple para construir la db
SYN = {
    # Farmacia / Medicamentos -- Todas las identificaremos como farmacias
    "farmacia": "farmacia",
    "medicamentos": "farmacia",
    "botiquin": "farmacia",
    "botiquín": "farmacia",
    "dispensario": "farmacia",
    "dispensacion": "farmacia",
    "dispensación": "farmacia",

    # Consulta con el medico general se escribe como consulta
    "consulta general": "consulta_general",
    "medico general": "consulta_general",
    "médico general": "consulta_general",
    "medicina general": "consulta_general",
    "consulta familiar": "consulta_general",
    #"consulta": "consulta_general",    # puede ser muy amplio; si genera ruido, eliminar o ver opciones
    "medico": "consulta_general",
    "medicos": "consulta_general",
    "revision" : "consulta_general",


    # Odontología
    "odontologia": "odontologia",
    "odontología": "odontologia",
    "dental": "odontologia",
    "dentista": "odontologia",
    "estomatologia": "odontologia",
    "estomatología": "odontologia",
    "extraccion": "odontologia",
    "extracción": "odontologia",
    "limpieza dental": "odontologia",
    "dentistas" : "odontologia",

    # Nutrición
    "nutricion": "nutricion",
    "nutrición": "nutricion",
    "nutriologo": "nutricion",
    "nutriólogo": "nutricion",
    "nutriologa": "nutricion",
    "nutrióloga": "nutricion",
    "clínica de nutrición": "nutricion",
    "clinica de nutricion": "nutricion",
    "nutricionista": "nutricion",
    "nutriologia": "nutricion",
    "consulta de nutricion": "nutricion",

    # Optometría
    "optometrista": "optometrista",
    "optometria": "optometrista",
    "optometría": "optometrista",
    "lentes": "optometrista",
    "examen de la vista": "optometrista",
    "vista": "optometrista",
    "optica": "optometrista",
    "óptica": "optometrista",
    "graduacion de lentes": "optometrista",
    "graduación de lentes": "optometrista",
    "examen visual": "optometrista",

    # Fisioterapia
    "fisioterapia": "fisioterapia",
    "fisioterapeuta": "fisioterapia",
    "rehabilitacion": "fisioterapia",
    "rehabilitación": "fisioterapia",
    "terapia fisica": "fisioterapia",
    "terapia física": "fisioterapia",

    # Psicología
    "psicologia": "psicologia",
    "psicología": "psicologia",
    "psicologo": "psicologia",
    "psicólogo": "psicologia",
    "psicologa": "psicologia",
    "psicóloga": "psicologia",
    "salud mental": "psicologia",
    "terapia psicologica": "psicologia",
    "terapia psicológica": "psicologia",

    # Mastografía sin dolor (x mi)
    "mastografia": "mastografia_sin_dolor",
    "mastografía": "mastografia_sin_dolor",
    "mamografia": "mastografia_sin_dolor",
    "mamografía": "mastografia_sin_dolor",
    "mastografia sin dolor": "mastografia_sin_dolor",
    "mastografía sin dolor": "mastografia_sin_dolor",
    "mamografia sin dolor": "mastografia_sin_dolor",
    "mamografía sin dolor": "mastografia_sin_dolor",
    "x mi": "mastografia_sin_dolor",
    "xmi": "mastografia_sin_dolor",

    # Consulta ginecológica
    "consulta ginecologica": "ginecologia",
    "consulta ginecológica": "ginecologia",
    "gineco": "ginecologia",
    "ginecologia": "ginecologia",
    "ginecología": "ginecologia",
    "ginecologo": "ginecologia",
    "ginecólogo": "ginecologia",
    "ginecologa": "ginecologia",
    "ginecóloga": "ginecologia",
    "papanicolaou": "ginecologia",
    "papanicolau": "ginecologia",
    "colposcopia": "ginecologia",
    "colposcopía": "ginecologia",

    # Laboratorio
    "laboratorio": "laboratorio",
    "analisis clinicos": "laboratorio",
    "análisis clínicos": "laboratorio",
    "examen de sangre": "laboratorio",
    "examenes": "laboratorio",
    "exámenes": "laboratorio",
    "quimica sanguinea": "laboratorio",
    "química sanguínea": "laboratorio",
    "biometria hematica": "laboratorio",
    "biometría hemática": "laboratorio",
    "pruebas de sangre": "laboratorio",

    ## Servicios extras o especiales, por UME
    "ozonoterapia": "ozonoterapia",
    "sesiones de ozono": "ozonoterapia",
    "terapia de ozono": "ozonoterapia",
    "ozono": "ozonoterapia",
    "fototerapia": "fototerapia",
    "hiperbarica": "hiperbarica",
    "camara hiperbarica": "hiperbarica",
    "cosmeatria": "cosmeatria",
    "estetica": "cosmeatria",
    "microneedling": "cosmeatria",
    "peeling": "cosmeatria",
    "mesoterapia": "cosmeatria",
    "botox": "cosmeatria",
    "acido hialuronico": "cosmeatria",
    "prp": "cosmeatria",
    "laserterapia": "laserterapia",
    "laser": "laserterapia",
    "terapia laser": "laserterapia",
}

SERVICIOS_CANONICOS = [
    "farmacia",
    "consulta_general",
    "odontologia",
    "nutricion",
    "optometrista",
    "fisioterapia",
    "psicologia",
    "mastografia_sin_dolor", # = mamografía sin dolor, x mi
    "ginecologia", # = gineco
    "laboratorio",
    "ozonoterapia",
    "fototerapia",
    "cosmeatria",
    "hiperbarica",
    "laserterapia"

]

SERVICE_LEXICON = { # Mas complejo para las query
# Servicio para direccionar a UME - Generalizacion de Terapias alternativas
  "ume_alternativas": {
    "syn": [
      "terapias alternativas", "terapias esteticas", "terapias estéticas",
      "unidad medica y estetica", "unidad médica y estética", "ume",
      "estetica avanzada", "cosmetologia", "cosmetología",
      "rehabilitacion avanzada", "terapias alternas",
      "medicina alternativa", "medicina alterna", "estetica",
      "cosmetica", "ozono", "sesiones de ozono", "terapia estetica",
      "ozonoterapia", "fototerapia", "cosmeatria", "hiperbarica", "laserterapia"
    ],
    "re": [
      r"\bterapias?\s+alternativ\w+\b",
      r"\bterapias?\s+est[ée]tic\w+\b",
      r"\bunidad\s+m[eé]dic[ae]\s+y\s+est[ée]tica\b|\bume\b",
      r"\bcosmetolog[íi]a\b",
      r"\brehabilitaci[óo]n\s+avanzad\w+\b",
      r"\bmedicin[a|as]?\s+alternativ\w+\b",
    ],
    "desc": "Búsqueda general de terapias alternativas/estéticas en UME."
  },

  "consulta_general": {
    "syn": [
      "consulta general","medico general","médico general","chequeo general", "doctor general"
      "revisión general","revision general","consulta con doctor","cita con doctor",
      "consulta médica","atencion medica","atención médica","primer contacto", "consulta medica",
        "medicos", "doctores", "revision general", "revision", "medico"
    ],
    "re":  [r"\b(consulta|chequeo|revisi[óo]n)\s+general\b", r"(\b(m[ée]dic[oa])|doctor)\s+general\b"],
    "desc": "Consulta médica de primer contacto y valoración general."
  },

  "odontologia": {
    "syn": [
      "dentista","odontologia","odontología","consulta dental","limpieza dental",
      "extraccion","extracción","caries","endodoncia","muela","diente","dolor de muela", "dentistas",
        "odontologos", "odontologo"
    ],
    "re":  [r"\b(dent(al|ista)|odontolog[ií]a|limpieza dental|caries|extracci[óo]n)\b"],
    "desc": "Atención dental: diagnóstico, limpieza, caries y procedimientos básicos."
  },

  "nutricion": {
    "syn": [
      "nutricion","nutrición","nutriologo","nutriólogo","dieta","plan alimenticio",
      "alimentacion","alimentación","bajar de peso","control de peso","obesidad", "nutriologos", "nutriologas"
    ],
    "re":  [r"\b(nutrici[óo]n|nutri[óo]log[oa]|dieta|alimentaci[óo]n)\b"],
    "desc": "Asesoría nutricional y planes de alimentación."
  },

  "optometrista": {
    "syn": [
      "optometrista","optometria","óptica","optica","lentes","graduacion de lentes",
      "examen de la vista","ojos","ver borroso","anteojos","armazon","montura", "lentes", "vista", "optometristas",
        "micas"
    ],
    "re":  [r"\b(optometr(ist(a|as)|ia)|[óo]ptic(a|as)|lentes|graduaci[óo]n|examen\s+de\s+la\s+vista|cambiar\s+micas)\b"],
    "desc": "Evaluación visual y graduación de lentes (óptica)."
  },

  "fisioterapia": {
    "syn": [
      "fisioterapia","terapia fisica","rehabilitacion","rehabilitación",
      "dolor muscular","esguince","terapias","fisio", "terapeuta", "fisio"
    ],
    "re":  [r"\b(fisioterapia|rehabilitaci[óo]n|terapia\s+f[ií]sica)\b"],
    "desc": "Rehabilitación y terapia física musculoesquelética."
  },

  "psicologia": {
    "syn": [
      "psicologia","psicología","psicologo","psicólogo","emocional", "psicológica"
      "ansiedad","estrés","estres","depresion","depresión","salud mental", "psicologos",
        "psicologas", "psicologa"
    ],
    "re":  [r"\b(psicol(og[ií]a|ogo|oga|ogos|ogas)|ansiedad|estr[ée]s|depresi[óo]n|psicol[ó|o]gica)\b"],
    "desc": "Atención psicológica y terapia."
  },

  "mastografia_sin_dolor": {
    "syn": [
      "mastografia","mastografía","mastografia sin dolor","x mi","cancer de mama","cáncer de mama",
      "mamografia","mamografía","estudio de mama"
    ],
    "re":  [r"\b(mastograf[ií]a|mamograf[ií]a|x\s?mi|mama)\b"],
    "desc": "Mastografía sin dolor (campaña X mi) para detección de cáncer de mama."
  },

  "ginecologia": {
    "syn": [
      "ginecologia","ginecología","ginecologo","ginecólogo","gineco","papanicolaou",
      "papanicolau","salud femenina","revision ginecologica", "ginecologa", "ginecologica"
    ],
    "re":  [r"\b(ginecolog[ií]a|ginec[óo]log[oa]|ginecol(ogica|ogos|ogas)|papanic(o|ó)l(a|á)u?)\b"],
    "desc": "Consulta ginecológica y salud de la mujer."
  },

  "laboratorio": {
    "syn": [
      "laboratorio","analisis","análisis","examen de sangre","pruebas","estudios",
      "toma de muestras","biometria hematica","biometría hemática","quimica sanguinea","química sanguínea",
    ],
    "re":  [r"\b(laboratorio|an[aá]lisis|ex[áa]menes?|pruebas|estudios)\b"],
    "desc": "Toma de muestras y análisis clínicos."
  },

  "farmacia": {
    "syn": [
      "medicamentos","farmacia","surtir receta","botiquin","botiquín","receta",
      "entrega de medicamentos", "farmacias", "medicinas", "medicina"
    ],
    "re":  [r"\b(medicamentos|farmacia|botiqu[ií]n|receta|venta\s+de\s+medicinas)\b"],
    "desc": "Surtido/entrega de medicamentos."
  },

 "ozonoterapia": {
    "syn": [
      "ozonoterapia", "ozono", "sesiones de ozono", "terapia de ozono",
      "oxigenoterapia"  # (muchos lo usan como sinónimo coloquial)
    ],
    "re": [
      r"\bozonoterap\w+\b",           # ozonoterapia / ozonoterapias
      r"\bozono\b",                   # ozono
      r"\b(terapia|sesiones?)\s+de\s+ozono\b",
      r"\boxigenoterap\w+\b"          # oxigenoterapia
    ],
    "desc": "Terapia con ozono (sesiones/valoración)."
  },

  "fototerapia": {
    "syn": [
      "fototerapia", "luz led", "fotobiomodulacion", "fotobiomodulación"
    ],
    "re": [
      r"\bfototerap\w+\b",
      r"\bluz\s+led\b",
      r"\bfotobiomodulaci[óo]n\b"
    ],
    "desc": "Fototerapia / luz LED para aplicaciones estéticas o clínicas."
  },

  "hiperbarica": {
    "syn": [
      "hiperbarica", "hiperbárica", "camara hiperbarica", "cámara hiperbárica",
      "oxigenoterapia hiperbarica", "oxigenoterapia hiperbárica"
    ],
    "re": [
      r"\bhiperbar(ic|ica)\b",
      r"\bc[aá]mara\s+hiperbar(ic|ica)\b",
      r"\boxigenoterapia\s+hiperbar(ic|ica)\b"
    ],
    "desc": "Cámara hiperbárica / oxigenoterapia hiperbárica."
  },

  "cosmeatria": {
    "syn": [
      "cosmeatria", "cosmeatra", "estetica", "estética", "terapias esteticas",
      "microneedling", "peeling", "peelings", "mesoterapia",
      "botox", "toxina botulinica", "toxina botulínica",
      "acido hialuronico", "ácido hialurónico",
      "prp", "plasma rico en plaquetas"
    ],
    "re": [
      r"\bcosmeatr\w+\b",                           # cosmeatria/cosmeátra
      r"\best[ée]tic\w+\b",                         # estética/esteticas
      r"\bmicroneedling\b",
      r"\bpeelings?\b",
      r"\bmesoterap\w+\b",
      r"\bbot[óo]x\b|\btoxina\s+botulinic\w+\b",
      r"\b[áa]cido\s+hialur[óo]nic\w+\b",
      r"\bprp\b|\bplasma\s+rico\s+en\s+plaquetas\b",
      r"\bterapias?\s+est[ée]tic\w+\b",
    ],
    "desc": "Cosmeatría/estética: microneedling, peelings, mesoterapia, bótox, ácido hialurónico, PRP."
  },

  "laserterapia": {
    "syn": [
      "laserterapia", "terapia laser", "terapia láser", "laser", "láser"
    ],
    "re": [
      r"\bl[aá]serterap\w+\b",
      r"\bterapia\s+l[áa]ser\b",
      r"\bl[áa]ser\b"
    ],
    "desc": "Terapias con láser (facial/corporal u otros usos)."
  },
}


# --- catálogo de estados (variantes comunes) ---
MX_STATES = {
    "aguascalientes": ["aguascalientes", "ags"],
    "baja california": ["baja california", "bc"],
    "baja california sur": ["baja california sur", "bcs"],
    "campeche": ["campeche"],
    "coahuila": ["coahuila", "coahuila de zaragoza", "coa", "coah"],
    "colima": ["colima"],
    "chiapas": ["chiapas"],
    "chihuahua": ["chihuahua", "chih"],
    "cdmx":
        ["cdmx","ciudad de mexico","d.f.","df","mexico city"],
    "durango": ["durango", "dgo"],
    "guanajuato": ["guanajuato", "gto", "gto."],
    "guerrero": ["guerrero", "gro"],
    "hidalgo": ["hidalgo", "hgo"],
    "jalisco": ["jalisco", "jal"],
    "estado de mexico":
        ["estado de mexico","edomex","e. de mexico","edomex","e de mexico","est de mexico",
        "mexico (edomex)", "edo mex", "mex", "edo de mex", "edo de mex", "mexico"],
    "michoacan": ["michoacan", "michoacan de ocampo", "mich"],
    "morelos": ["morelos"],
    "nayarit": ["nayarit", "nay"],
    "nuevo leon": ["nuevo leon","nl","nvo leon","n. leon","nuevoleon"],
    "oaxaca": ["oaxaca", "oax", "oax."],
    "puebla": ["puebla","pue"],
    "queretaro": ["queretaro", "qro", "queretaro de arteaga"],
    "quintana roo": ["quintana roo", "qroo"],
    "san luis potosi": ["san luis potosi", "slp", "san luis"],
    "sinaloa": ["sinaloa", "sin"],
    "sonora": ["sonora", "son"],
    "tabasco": ["tabasco", "tab"],
    "tamaulipas": ["tamaulipas", "tamps"],
    "tlaxcala": ["tlaxcala", "tlax"],
    "veracruz": ["veracruz", "ver", "veracruz de ignacio de la llave"],
    "yucatan": ["yucatan", "yuc"],
    "zacatecas": ["zacatecas", "zac"]
}


"""
Aliases para municipios, variantes
"""

STOPWORDS = {"de","del","de los","de las","la","las","los","el","y","a"}
def gen_variants(canonical: str) -> set[str]:
    """Genera variantes simples útiles para matching de alias."""
    s = norm_txt(canonical)
    out = {s}
    toks = [t for t in s.split() if t not in STOPWORDS]
    if not toks:
        return out
    # unión sin espacios (p.ej. 'vcarranza')
    out.add(''.join(toks))
    # inicial + resto (p.ej. 'v carranza', 'v. carranza')
    if len(toks) >= 2:
        out.add(toks[0][0] + ' ' + ' '.join(toks[1:]))
        out.add(toks[0][0] + '. ' + ' '.join(toks[1:]))
        out.add(toks[0][0] + ''.join(toks[1:]))
    return out

def build_muni_aliases_from_catalog(muni_by_state: dict[str, list[str]]) -> dict[str, tuple[str,str]]:
    """
    Devuelve dict alias_norm -> (municipio_canon_norm, estado_canon_norm)
    a partir de MUNI_BY_STATE_MINI.
    """
    out = {}
    for state, munis in muni_by_state.items():
        st = norm_txt(state)
        for m in munis:
            m_norm = norm_txt(m)
            for alias in gen_variants(m_norm):
                out[alias] = (m_norm, st)
    return out

def merge_aliases(auto_aliases: dict, manual_aliases: dict) -> dict:
    """Manual > Auto (los manuales sobrescriben)."""
    merged = dict(auto_aliases)
    for k, v in manual_aliases.items():
        merged[norm_txt(k)] =(norm_txt(v[0]), norm_txt(v[1]))
    return merged


NUEVO_LEON_ALIASES = {
    "monterrey": ("monterrey", "nuevo leon"),
    "san pedro garza garcia": ("san pedro garza garcia", "nuevo leon"),
    "guadalupe": ("guadalupe", "nuevo leon"),
    "apodaca": ("apodaca", "nuevo leon"),
    "san nicolas de los garza": ("san nicolas de los garza", "nuevo leon"),
    "santa catarina": ("santa catarina", "nuevo leon"),
}

SAN_LUIS_POTOSI_ALIASES = {
    # Zona metropolitana
    "san luis potosi": ("san luis potosi", "san luis potosi"),
    "slp": ("san luis potosi", "san luis potosi"),
    "soledad": ("soledad de graciano sanchez", "san luis potosi"),
    "soledad de graciano sanchez": ("soledad de graciano sanchez", "san luis potosi"),
    "graciano sanchez": ("soledad de graciano sanchez", "san luis potosi"),

    # Huasteca
    "ciudad valles": ("ciudad valles", "san luis potosi"),
    "cd valles": ("ciudad valles", "san luis potosi"),
    "valles slp": ("ciudad valles", "san luis potosi"),
    "tamuin": ("tamuin", "san luis potosi"),
    "tamazunchale": ("tamazunchale", "san luis potosi"),
    "aquismon": ("aquismon", "san luis potosi"),
    "aquismón": ("aquismon", "san luis potosi"),
    "xilitla": ("xilitla", "san luis potosi"),

    # Zona Media
    "rioverde": ("rioverde", "san luis potosi"),
    "rio verde": ("rioverde", "san luis potosi"),
    "cd fernandez": ("ciudad fernandez", "san luis potosi"),
    "ciudad fernandez": ("ciudad fernandez", "san luis potosi"),

    # Altiplano
    "matehuala": ("matehuala", "san luis potosi"),
    "cedral": ("cedral", "san luis potosi"),
    "charcas": ("charcas", "san luis potosi"),
    "venado": ("venado", "san luis potosi"),
    "villa de ramos": ("villa de ramos", "san luis potosi"),

    # Otras cabeceras relevantes
    "ciudad del maiz": ("ciudad del maiz", "san luis potosi"),
    "ciudad del maíz": ("ciudad del maiz", "san luis potosi"),
    "ebano": ("ebano", "san luis potosi"),
    "ébano": ("ebano", "san luis potosi"),
    "coxcatlan": ("coxcatlan", "san luis potosi"),
    "coxcatlán": ("coxcatlan", "san luis potosi"),
    "huehuetlan": ("huehuetlan", "san luis potosi"),
    "huehuetlán": ("huehuetlan", "san luis potosi"),
    "tanlajas": ("tanlajas", "san luis potosi"),

    "villa de reyes": ("villa de reyes", "san luis potosi"),
    "san luis rey": ("san luis rey", "san luis potosi"),
    "providencia": ("providencia", "san luis potosi"),

    "barrio de tlaxcala": ("barrio de tlaxcala", "san luis potosi"),


    "ciudad satelite": ("san luis potosi", "san luis potosi"),  # barrio que a veces aparece como municipio
}
# --- Alcaldías CDMX (canónicas, sin acento) ---
CDMX_BOROUGHS = {
# Álvaro Obregón
    "alvaro obregon": ("alvaro obregon","cdmx"),
    "a obregon": ("alvaro obregon","cdmx"),
    "ao": ("alvaro obregon","cdmx"),
    # Azcapotzalco
    "azcapotzalco": ("azcapotzalco","cdmx"),
    "azca": ("azcapotzalco","cdmx"),
    # Benito Juárez
    "benito juarez": ("benito juarez","cdmx"),
    "benito juares": ("benito juarez","cdmx"),
    "bj": ("benito juarez","cdmx"),
    # Coyoacán
    "coyoacan": ("coyoacan","cdmx"),
    # Cuajimalpa
    "cuajimalpa": ("cuajimalpa de morelos","cdmx"),
    "cuajimalpa de morelos": ("cuajimalpa de morelos","cdmx"),
    # Cuauhtémoc
    "cuauhtemoc": ("cuauhtemoc","cdmx"),
    "Roma" : ("cuauhtemoc","cdmx"),
    # Gustavo A. Madero
    "gustavo a madero": ("gustavo a. madero","cdmx"),
    "gustavo a. madero": ("gustavo a. madero","cdmx"),
    "g a madero": ("gustavo a. madero","cdmx"),
    "gam": ("gustavo a. madero","cdmx"),
    # Iztacalco
    "iztacalco": ("iztacalco","cdmx"),
    # Iztapalapa
    "iztapalapa": ("iztapalapa","cdmx"),
    "izta": ("iztapalapa","cdmx"),
    # Magdalena Contreras
    "magdalena contreras": ("magdalena contreras","cdmx"),
    # Milpa Alta
    "milpa alta": ("milpa alta","cdmx"),
    # Tláhuac
    "tlahuac": ("tlahuac","cdmx"),
    # Tlalpan
    "tlalpan": ("tlalpan","cdmx"),
    # Venustiano Carranza
    "venustiano carranza": ("venustiano carranza","cdmx"),
    "v carranza": ("venustiano carranza","cdmx"),
    "vcarranza": ("venustiano carranza","cdmx"),
    "vc": ("venustiano carranza","cdmx"),
    # Xochimilco
    "xochimilco": ("xochimilco","cdmx"),
    # Miguel Hidalgo
    "miguel hidalgo": ("miguel hidalgo","cdmx"),
    "mh": ("miguel hidalgo","cdmx"),
}

# --- Mini-catálogo de cada Estado y municipios (municipios canónicos frecuentes) ---
EDO_MX_BOROUGHS = {
        # Ecatepec
    "ecatepec": ("ecatepec de morelos","estado de mexico"),
    "ecatepec de morelos": ("ecatepec de morelos","estado de mexico"),
    # Naucalpan
    "naucalpan": ("naucalpan de juarez","estado de mexico"),
    "naucalpan de juarez": ("naucalpan de juarez","estado de mexico"),
    "nauc": ("naucalpan de juarez","estado de mexico"),
    # Tlalnepantla
    "tlalnepantla": ("tlalnepantla de baz","estado de mexico"),
    "tlalnepantla de baz": ("tlalnepantla de baz","estado de mexico"),
    "tlalne": ("tlalnepantla de baz","estado de mexico"),
    # Nezahualcóyotl
    "neza": ("nezahualcoyotl","estado de mexico"),
    "cd neza": ("nezahualcoyotl","estado de mexico"),
    "nezahualcoyotl": ("nezahualcoyotl","estado de mexico"),
    # Toluca / Metepec
    "toluca": ("toluca de lerdo","estado de mexico"),
    "toluca de lerdo": ("toluca de lerdo","estado de mexico"),
    "metepec": ("metepec","estado de mexico"),
    # Atizapán
    "atizapan": ("atizapan de zaragoza","estado de mexico"),
    "atizapan de zaragoza": ("atizapan de zaragoza","estado de mexico"),
    # Coacalco
    "coacalco": ("coacalco de berriozabal","estado de mexico"),
    "coacalco de berriozabal": ("coacalco de berriozabal","estado de mexico"),
    # Izcalli
    "izcalli": ("cuautitlan izcalli","estado de mexico"),
    "cuautitlan izcalli": ("cuautitlan izcalli","estado de mexico"),
    "c izcalli": ("cuautitlan izcalli","estado de mexico"),
    # Tecámac
    "tecamac": ("tecamac","estado de mexico"),
    # Chalco
    "chalco": ("chalco","estado de mexico"),
    # Chimalhuacán
    "chimalhuacan": ("chimalhuacan","estado de mexico"),
    # Texcoco
    "texcoco": ("texcoco","estado de mexico"),
    # Ixtapaluca
    "ixtapaluca": ("ixtapaluca","estado de mexico"),
    # Tultitlán
    "tultitlan": ("tultitlan","estado de mexico"),
    # Nicolás Romero
    "nicolas romero": ("nicolas romero","estado de mexico"),
    # Huixquilucan
    "huixquilucan": ("huixquilucan","estado de mexico"),
    # Zumpango
    "zumpango": ("zumpango","estado de mexico"),
    # Valle de Bravo
    "valle de bravo": ("valle de bravo","estado de mexico"),
    # Lerma
    "lerma": ("lerma","estado de mexico"),
    "lerma de villada": ("lerma de villada","estado de mexico"),
    # San Mateo Atenco
    "san mateo atenco": ("san mateo atenco","estado de mexico"),
    # Teotihuacán
    "teotihuacan": ("teotihuacan","estado de mexico"),
    # Teoloyucan
    "teoloyucan": ("teoloyucan","estado de mexico"),
    # Melchor Ocampo
    "melchor ocampo": ("melchor ocampo","estado de mexico"),
    # Tultepec
    "tultepec": ("tultepec","estado de mexico"),
    # La Paz / Los Reyes La Paz
    "la paz": ("la paz","estado de mexico"),
    "los reyes la paz": ("los reyes la paz","estado de mexico"),
    # Cuautitlán (cabecera)
    "cuautitlan": ("cuautitlan","estado de mexico"),
    # Tenango / Tenancingo / Tejupilco (valle del sur)
    "tenango del valle": ("tenango del valle","estado de mexico"),
    "tenancingo": ("tenancingo","estado de mexico"),
    "tejupilco": ("tejupilco","estado de mexico"),
    "tejupilco de hidalgo": ("tejupilco de hidalgo", "estado de mexico"),
    # TepoTZOTLÁN (ojo ortografía)
    "tepotzotlan": ("tepotzotlan","estado de mexico"),
    # SAN VICENTE CHICOLOAPAN DE JUÁREZ
    "san vicente chicoloapan de juarez" : ("chicoloapan de juarez", "estado de mexico"),
    "chicoloapan": ("chicoloapan de juarez","estado de mexico"),
    "chicoloapan de juarez": ("chicoloapan de juarez","estado de mexico"),
    "san vicente" : ("san vicente chicoloapan de juarez","estado de mexico"),
    # JILOTEPEC DE MOLINA ENRÍQUEZ
    "jilotepec de molina enriquez" : ("jilotepec de molina enriquez","estado de mexico"),
    "jilotepec" : ("jilotepec de molina enriquez","estado de mexico"),

    # Tonanitla
    "tonanitla": ("tonanitla","estado de mexico"),

    #Polotitlán de la Ilustración
    "polotitlan de la ilustracion": ("polotitlan de la ilustracion", "estado de mexico"),

    "san isidro rayon" : ("san isidro rayon","estado de mexico"),
    "rayon": ("san isidro rayon","estado de mexico"),

    "lomas de san sebastian": ("lomas de san sebastian", "estado de mexico"),

}

YUCATAN_ALIASES = {
    "merida": ("merida", "yucatan"),
    "kanasin": ("kanasin", "yucatan"),
    "progreso": ("progreso", "yucatan"),
    "valladolid": ("valladolid", "yucatan"),
    "tizimin": ("tizimin", "yucatan"),
    "motul": ("motul", "yucatan"),
    "izamal": ("izamal", "yucatan"),
    "tekax": ("tekax", "yucatan"),
    "uman": ("uman", "yucatan"),
    "ticul": ("ticul", "yucatan"),
    "peto": ("peto", "yucatan"),

}

AGS_ALIASES = {
    # Aguascalientes capital
    "aguascalientes": ("aguascalientes", "aguascalientes"),
    "aguascalientes capital": ("aguascalientes", "aguascalientes"),
    "ags": ("aguascalientes", "aguascalientes"),

    # Jesús María
    "jesus maria": ("jesus maria", "aguascalientes"),
    "jesús maría": ("jesus maria", "aguascalientes"),

    # Calvillo
    "calvillo": ("calvillo", "aguascalientes"),

    # Pabellón de Arteaga
    "pabellon de arteaga": ("pabellon de arteaga", "aguascalientes"),
    "pabellón de arteaga": ("pabellon de arteaga", "aguascalientes"),

    # Rincón de Romos
    "rincon de romos": ("rincon de romos", "aguascalientes"),
    "rincón de romos": ("rincon de romos", "aguascalientes"),

    # San Francisco de los Romo
    "san francisco de los romo": ("san francisco de los romo", "aguascalientes"),
    "sfr": ("san francisco de los romo", "aguascalientes"),

    # Tepezalá
    "tepezala": ("tepezala", "aguascalientes"),
    "tepezalá": ("tepezala", "aguascalientes"),

    # Cosío
    "cosio": ("cosio", "aguascalientes"),
    "cosío": ("cosio", "aguascalientes"),

    # Asientos
    "asientos": ("asientos", "aguascalientes"),

    # San José de Gracia
    "san jose de gracia": ("san jose de gracia", "aguascalientes"),
    "san josé de gracia": ("san jose de gracia", "aguascalientes"),

    # El Llano
    "el llano": ("el llano", "aguascalientes"),

    # Santa Elena
    "santa elena": ("santa elena", "aguascalientes"),

    "villa juarez": ("villa juarez", "aguascalientes"),
}

OAXACA_ALIASES = {
    "oaxaca": ("oaxaca de juarez", "oaxaca"),
    "oaxaca de juarez": ("oaxaca de juarez", "oaxaca"),
    "huatulco": ("santa maria huatulco", "oaxaca"),
    "santa maria huatulco": ("santa maria huatulco", "oaxaca"),
    "salina cruz": ("salina cruz", "oaxaca"),
    "tehuantepec": ("santo domingo tehuantepec", "oaxaca"),
    "santo domingo tehuantepec": ("santo domingo tehuantepec", "oaxaca"),
    "juchitan": ("juchitan de zaragoza", "oaxaca"),
    "juchitan de zaragoza": ("juchitan de zaragoza", "oaxaca"),
    "tlaxiaco": ("heroica ciudad de tlaxiaco", "oaxaca"),
    "heroica ciudad de tlaxiaco": ("heroica ciudad de tlaxiaco", "oaxaca"),
    "huautla de jimenez" : ("huautla de jimenez", "oaxaca"),
    "huautla" : ("huautla de jimenez", "oaxaca"),
    "san pedro pochutla" : ("san pedro pochutla", "oaxaca"),
    "pochutla" : ("san pedro pochutla", "oaxaca"),
    "putla": ("putla villa de guerrero", "oaxaca"),
    "putla villa de guerrero": ("putla villa de guerrero", "oaxaca"),
    "villa De guerrero" : ("putla villa De guerrero", "oaxaca"),
    "tuxtepec": ("san juan bautista tuxtepec", "oaxaca"),
    "san juan bautista tuxtepec": ("san juan bautista tuxtepec", "oaxaca"),
    "mitla": ("san pablo villa de mitla", "oaxaca"),
    "san pablo villa de mitla": ("san pablo villa de mitla", "oaxaca"),
    "pinotepa": ("pinotepa nacional", "oaxaca"),
    "pinotepa nacional": ("pinotepa nacional", "oaxaca"),
    "nochixtlan": ("asuncion nochixtlan", "oaxaca"),
    "asuncion nochixtlan": ("asuncion nochixtlan", "oaxaca"),
    "huajuapan": ("heroica ciudad de huajuapan de leon", "oaxaca"),
    "huajuapan de leon": ("heroica ciudad de huajuapan de leon", "oaxaca"),
    "tlacolula": ("tlacolula de matamoros", "oaxaca"),
    "tlacolula de matamoros": ("tlacolula de matamoros", "oaxaca"),
    "santiago juxtlahuaca": ("santiago juxtlahuaca", "oaxaca"),
    "juxtlahuaca": ("santiago juxtlahuaca", "oaxaca"),
    "santa cruz xoxocotlan": ("santa cruz xoxocotlan", "oaxaca"),
    "santa lucia del camino": ("santa lucia del camino", "oaxaca"),
    "miahuatlan de porfirio diaz": ("miahuatlan de porfirio diaz", "oaxaca"),
    "miahuatlan": ("miahuatlan de porfirio diaz", "oaxaca"),
    "acatlan de perez figueroa": ("acatlan de perez figueroa", "oaxaca"),
}

GUERRERO_ALIASES = {
    "acapulco": ("acapulco de juarez", "guerrero"),
    "acapulco de juarez": ("acapulco de juarez", "guerrero"),
    "chilpancingo": ("chilpancingo de los bravo", "guerrero"),
    "chilpancingo de los bravo": ("chilpancingo de los bravo", "guerrero"),
    "iguala": ("iguala de la independencia", "guerrero"),
    "iguala de la independencia": ("iguala de la independencia", "guerrero"),
    "taxco": ("taxco de alarcon", "guerrero"),
    "taxco de alarcon": ("taxco de alarcon", "guerrero"),
    "zihuatanejo": ("zihuatanejo de azueta", "guerrero"),
    "zihuatanejo de azueta": ("zihuatanejo de azueta", "guerrero"),
    "tierra colorada": ("juan r escudero", "guerrero"),
    "chilapa": ("chilapa de alvarez", "guerrero"),
    "chilapa de alvarez": ("chilapa de alvarez", "guerrero"),
    "ayutla": ("ayutla de los libres", "guerrero"),
    "ayutla de los libres": ("ayutla de los libres", "guerrero"),
    "tixtla": ("tixtla de guerrero", "guerrero"),
    "tixtla de guerrero": ("tixtla de guerrero", "guerrero"),
    "coyuca": ("coyuca de benitez", "guerrero"),
    "coyuca de benitez": ("coyuca de benitez", "guerrero"),
    "ometepec": ("ometepec", "guerrero"),
    "tecpan": ("tecpan de galeana", "guerrero"),
    "tecpan de galeana": ("tecpan de galeana", "guerrero"),
    "arcelia": ("arcelia", "guerrero"),
    "atoyac": ("atoyac de alvarez", "guerrero"),
    "atoyac de alvarez": ("atoyac de alvarez", "guerrero"),
}

HIDALGO_ALIASES = {
    # Pachuca
    "pachuca": ("pachuca de soto", "hidalgo"),
    "pachuca de soto": ("pachuca de soto", "hidalgo"),
    "pachuca hgo": ("pachuca de soto", "hidalgo"),

    # Tulancingo
    "tulancingo": ("tulancingo de bravo", "hidalgo"),
    "tulancingo de bravo": ("tulancingo de bravo", "hidalgo"),

    # Tizayuca
    "tizayuca": ("tizayuca", "hidalgo"),

    # Tula
    "tula": ("tula de allende", "hidalgo"),
    "tula de allende": ("tula de allende", "hidalgo"),
    "tula hgo": ("tula de allende", "hidalgo"),

    # Huejutla
    "huejutla": ("huejutla de reyes", "hidalgo"),
    "huejutla de reyes": ("huejutla de reyes", "hidalgo"),

    # Actopan
    "actopan": ("actopan", "hidalgo"),

    # Ixmiquilpan
    "ixmiquilpan": ("ixmiquilpan", "hidalgo"),

    # Tepeji del Río
    "tepeji": ("tepeji del río de ocampo", "hidalgo"),
    "tepeji del rio": ("tepeji del río de ocampo", "hidalgo"),
    "tepeji del río": ("tepeji del río de ocampo", "hidalgo"),

    # Apan
    "apan": ("apan", "hidalgo"),

    # Zacualtipán
    "zacualtipan": ("zacualtipan de angeles", "hidalgo"),
    "zacualtipan de angeles": ("zacualtipan de angeles", "hidalgo"),
    "zacualtipan de ángeles": ("zacualtipan de angeles", "hidalgo"),

    # Tepeapulco
    "tepeapulco": ("tepeapulco", "hidalgo"),
    "cd sahagun": ("tepeapulco", "hidalgo"),
    "ciudad sahagun": ("tepeapulco", "hidalgo"),
}

CHIHUAHUA_ALIASES = {
    # Chihuahua
    "chihuahua": ("chihuahua", "chihuahua"),
    "chih": ("chihuahua", "chihuahua"),
    "chihuahua capital": ("chihuahua", "chihuahua"),

    # Ciudad Juárez
    "juarez": ("ciudad juarez", "chihuahua"),
    "ciudad juarez": ("ciudad juarez", "chihuahua"),
    "cd juarez": ("ciudad juarez", "chihuahua"),

    # Delicias
    "delicias": ("delicias", "chihuahua"),

    # Parral
    "parral": ("parral", "chihuahua"),
    "hidalgo del parral": ("parral", "chihuahua"),

    # Nuevo Casas Grandes
    "nuevo casas grandes": ("nuevo casas grandes", "chihuahua"),
    "cd casas grandes": ("nuevo casas grandes", "chihuahua"),
    "casas grandes": ("nuevo casas grandes", "chihuahua"),

    # Jiménez
    "jimenez": ("jimenez", "chihuahua"),
    "ciudad jimenez": ("jimenez", "chihuahua"),

    # Camargo
    "camargo": ("camargo", "chihuahua"),

    # Meoqui
    "meoqui": ("meoqui", "chihuahua"),

    # Aldama
    "aldama": ("aldama", "chihuahua"),
}

COAHUILA_ALIASES = {
    # ZM Saltillo–Ramos Arizpe–Arteaga–Gral. Cepeda
    "saltillo": ("saltillo", "coahuila"),
    "ramos arizpe": ("ramos arizpe", "coahuila"),
    "arteaga": ("arteaga", "coahuila"),
    "general cepeda": ("general cepeda", "coahuila"),
    "gral cepeda": ("general cepeda", "coahuila"),

    # ZM La Laguna (lado Coahuila)
    "torreon": ("torreon", "coahuila"),
    "matamoros coah": ("matamoros", "coahuila"),
    "matamoros (coahuila)": ("matamoros", "coahuila"),
    "matamoros coahuila": ("matamoros", "coahuila"),
    "viesca": ("viesca", "coahuila"),
#    "san pedro": ("san pedro", "coahuila"),
    "san pedro de las colonias": ("san pedro", "coahuila"),
    "francisco i madero": ("francisco i. madero", "coahuila"),
    "francisco i. madero": ("francisco i. madero", "coahuila"),
    "fim coahuila": ("francisco i. madero", "coahuila"),

    # Centro–Carbonífera–Norte
    "monclova": ("monclova", "coahuila"),
    "frontera": ("frontera", "coahuila"),
    "castanos": ("castanos", "coahuila"),
    "castaños": ("castanos", "coahuila"),
    "candela": ("candela", "coahuila"),
    "naderadores": ("nadadores", "coahuila"),   # por si hay typo frecuente
    "nadadores": ("nadadores", "coahuila"),
    "sacramento": ("sacramento", "coahuila"),
    "lamadrid": ("lamadrid", "coahuila"),
    "cuatro cienegas": ("cuatro cienegas", "coahuila"),
    "cuatrociénegas": ("cuatro cienegas", "coahuila"),
    "cuatro cienegas de carranza": ("cuatro cienegas", "coahuila"),
    "parras": ("parras de la fuente", "coahuila"),
    "parras de la fuente": ("parras de la fuente", "coahuila"),
    "sabinas": ("sabinas", "coahuila"),
    "san juan de sabinas": ("san juan de sabinas", "coahuila"),
    "nueva rosita": ("san juan de sabinas", "coahuila"),
    "melchor muzquiz": ("melchor muzquiz", "coahuila"),
    "melchor mùzquiz": ("melchor muzquiz", "coahuila"),
    "muzquiz": ("melchor muzquiz", "coahuila"),

    # Norte fronteriza
    "piedras negras": ("piedras negras", "coahuila"),
    "cd piedras negras": ("piedras negras", "coahuila"),
    "ciudad piedras negras": ("piedras negras", "coahuila"),
    "acuna": ("acuna", "coahuila"),
    "acuña": ("acuna", "coahuila"),
    "ciudad acuna": ("acuna", "coahuila"),
    "ciudad acuña": ("acuna", "coahuila"),
    "cd acuna": ("acuna", "coahuila"),
    "cd. acuna": ("acuna", "coahuila"),
    "cd acuña": ("acuna", "coahuila"),
    "allende coah": ("allende", "coahuila"),
    "allende": ("allende", "coahuila"),
    # "morelos coah": ("morelos", "coahuila"),
    # "morelos": ("morelos", "coahuila"),
    "nava": ("nava", "coahuila"),
    "zaragoza coah": ("zaragoza", "coahuila"),
    "zaragoza": ("zaragoza", "coahuila"),
    "villa union": ("villa union", "coahuila"),
    "villa unión": ("villa union", "coahuila"),
    # "guerrero coah": ("guerrero", "coahuila"),
    # "guerrero (coahuila)": ("guerrero", "coahuila"),
    # "hidalgo coah": ("hidalgo", "coahuila"),
    # "hidalgo (coahuila)": ("hidalgo", "coahuila"),
    "jimenez coah": ("jimenez", "coahuila"),
    "jimenez (coahuila)": ("jimenez", "coahuila"),
    "progreso coah": ("progreso", "coahuila"),
    "progreso (coahuila)": ("progreso", "coahuila"),
    "ocampo coah": ("ocampo", "coahuila"),
    "ocampo (coahuila)": ("ocampo", "coahuila"),
    "sierra mojada": ("sierra mojada", "coahuila"),
    "escobedo coah": ("escobedo", "coahuila"),
    "escobedo (coahuila)": ("escobedo", "coahuila"),
}

TLAXCALA_ALIASES = {
    "tlaxcala de xicohtencatl": ("tlaxcala de xicohtencatl", "tlaxcala"),
    "tlaxcala" : ("tlaxcala de xicohtencatl", "tlaxcala"),
    "tlax": ("tlaxcala de xicohtencatl", "tlaxcala"),
    "apizaco": ("apizaco", "tlaxcala"),
    "teolocholco": ("teolocholco", "tlaxcala"),
    "santa cruz quilehtla": ("santa cruz quilehtla", "tlaxcala"),
    "teotlalpan": ("teotlalpan", "tlaxcala"),
    "tetla" : ("tetla", "tlaxcala"),
    "contla de juan cuamatzi": ("contla de juan cuamatzi", "tlaxcala"),
    "contra de juan": ("contra de juan cuamatzi", "tlaxcala"),
    "calpulalpan": ("calpulalpan", "tlaxcala"),
    "calpulalpan tlaxcala": ("calpulalpan", "tlaxcala"),
    "ixtacuixtla de mariano matamoros": ("ixtacuixtla de mariano matamoros", "tlaxcala"),
    "ixtacuixtla": ("ixtacuixtla de mariano matamoros", "tlaxcala"),
    "tlaxco": ("tlaxco", "tlaxcala"),
    "yauhquemehcan": ("yauhquemehcan", "tlaxcala"),
    "chiautempan" : ("chiautempan", "tlaxcala"),
    "huamantla" : ("huamantla", "tlaxcala"),
    "papalotla de xicohtencatl" : ("papalotla de xicohtencatl", "tlaxcala"),
    "papalotla": ("papalotla de xicohtencatl", "tlaxcala"),
    "nacamilpa" : ("nacamilpa", "tlaxcala"),
    "san pablo del monte": ("san pablo del monte", "tlaxcala"),
    "pablo del monte": ("san pablo del monte", "tlaxcala"),
    "atepatitlan": ("san pablo atepatitlan", "tlaxcala"),
    "zacatelco": ("zacatelco", "tlaxcala"),
}

JALISCO_ALIASES = {
    "guadalajara": ("guadalajara", "jalisco"),
    "zapopan": ("zapopan", "jalisco"),
    "tonala": ("tonala", "jalisco"),
    "tlaquepaque": ("san pedro tlaquepaque", "jalisco"),
    "san pedro tlaquepaque": ("san pedro tlaquepaque", "jalisco"),
    "tlajomulco": ("tlajomulco de zuñiga", "jalisco"),
    "tlajomulco de zuniga": ("tlajomulco de zuñiga", "jalisco"),
    "tlajomulco de zuñiga": ("tlajomulco de zuñiga", "jalisco"),
    "puerto vallarta": ("puerto vallarta", "jalisco"),
    "tequila": ("tequila", "jalisco"),
    "arandas": ("arandas", "jalisco"),
    "lagos de moreno": ("lagos de moreno", "jalisco"),
    "autlan": ("autlan de navarro", "jalisco"),
    "autlan de navarro": ("autlan de navarro", "jalisco"),
    "cihuatlan": ("cihuatlan", "jalisco"),
    "el salto": ("el salto", "jalisco"),
    "atotonilco el alto": ("atotonilco el alto", "jalisco"),
    "jamay": ("jamay", "jalisco"),
    "chapala": ("chapala", "jalisco"),
    "ocotlan": ("ocotlan", "jalisco"),
    "tamazula": ("tamazula de gordiano", "jalisco"),
    "tamazula de gordiano": ("tamazula de gordiano", "jalisco"),
    "ixtapa": ("ixtapa", "jalisco"),
}

SINALOA_ALIASES = {
    "culiacan": ("culiacan de rosales", "sinaloa"),
    "culiacan de rosales": ("culiacan de rosales", "sinaloa"),
    "mazatlan": ("mazatlan", "sinaloa"),
    "los mochis": ("ahome", "sinaloa"),
    "ahome": ("ahome", "sinaloa"),
    "guasave": ("guasave", "sinaloa"),
    "navolato": ("navolato", "sinaloa"),
    "salvador alvarado": ("salvador alvarado", "sinaloa"),
    "guamuchil": ("salvador alvarado", "sinaloa"),
    "el fuerte": ("el fuerte", "sinaloa"),
    "angostura": ("angostura", "sinaloa"),
    "mocorito": ("mocorito", "sinaloa"),
    "cosala": ("cosala", "sinaloa"),
    "rosario": ("rosario", "sinaloa"),
    "escuinapa de hidalgo": ("escuinapa de hidalgo", "sinaloa"),
    "escuinapa" : ("escuinapa de hidalgo", "sinaloa"),
}

DURANGO_ALIASES = {
    "durango": ("victoria de durango", "durango"),
    "dgo": ("victoria de durango", "durango"),
    "victoria de durango": ("victoria de durango", "durango"),
    "gomez palacio": ("gomez palacio", "durango"),
    "lerdo": ("lerdo", "durango"),
    "canatlan": ("canatlan", "durango"),
    "nombre de dios": ("nombre de dios", "durango"),
    "el salto": ("pueblo nuevo", "durango"),
    "santiago papasquiaro": ("santiago papasquiaro", "durango"),
    "guadalupe victoria": ("guadalupe victoria", "durango"),
    "poanas": ("poanas", "durango"),
    "mezquital": ("mezquital", "durango"),
}

GUANAJUATO_ALIASES = {
    "leon": ("leon de los aldama", "guanajuato"),
    "leon de los aldama": ("leon de los aldama", "guanajuato"),
    "irapuato": ("irapuato", "guanajuato"),
    "celaya": ("celaya", "guanajuato"),
    "guanajuato": ("guanajuato", "guanajuato"),
    "salamanca": ("salamanca", "guanajuato"),
    "silao": ("silao de la victoria", "guanajuato"),
    "silao de la victoria": ("silao de la victoria", "guanajuato"),
    "san miguel de allende": ("san miguel de allende", "guanajuato"),
    "dolores hidalgo": ("dolores hidalgo cuna de la independencia nacional", "guanajuato"),
    "dolores hidalgo cuna de la independencia nacional": ("dolores hidalgo cuna de la independencia nacional", "guanajuato"),
    "acambaro": ("acambaro", "guanajuato"),
    "pénjamo": ("penjamo", "guanajuato"),
    "penjamo": ("penjamo", "guanajuato"),
    "san felipe": ("san felipe", "guanajuato"),
    "purisima del rincon": ("purisima del rincon", "guanajuato"),
    "moroleon": ("moroleon", "guanajuato"),
    "uriangato": ("uriangato", "guanajuato"),
    "tarimoro": ("tarimoro", "guanajuato"),
    "coroneo": ("coroneo", "guanajuato"),
    "apaseo el grande": ("apaseo el grande", "guanajuato"),
}

CAMPECHE_ALIASES = {
    "san francisco de campeche": ("san francisco de campeche", "campeche"),
    "campeche": ("san francisco de campeche", "campeche"),
    "carmen": ("ciudad del carmen", "campeche"),
    # "ciudad del carmen": ("ciudad del carmen", "campeche"),
    "escárcega": ("escarcega", "campeche"),
    "escarcega": ("escarcega", "campeche"),
    "champoton": ("champoton", "campeche"),
    "hool": ("hool", "campeche"),
    "tenabo": ("tenabo", "campeche"),
    "hecelchakan": ("hecelchakan", "campeche"),
    "palizada": ("palizada", "campeche"),
    "calakmul": ("calakmul", "campeche"),
    "candelaria": ("candelaria", "campeche"),
}

PUEBLA_ALIASES = {
    "puebla": ("puebla","puebla"),
    "san andres cholula": ("san andres cholula","puebla"),
    "san pedro cholula": ("san pedro cholula","puebla"),
    "tehuacan": ("tehuacan","puebla"),
}

TAMAULIPAS_ALIASES = {
    "reynosa": ("reynosa","tamaulipas"),
}

# Mapa de alias -> municipio canónico (por estado si aplica)
MUNICIPALITY_ALIASES = {
    # CDMX variantes frecuentes #######################################################
    **CDMX_BOROUGHS,
    #################################################################
    **NUEVO_LEON_ALIASES,
    **SAN_LUIS_POTOSI_ALIASES,
    **JALISCO_ALIASES,
    **SINALOA_ALIASES,
    **GUANAJUATO_ALIASES,
    **DURANGO_ALIASES,
    **CAMPECHE_ALIASES,
    **CHIHUAHUA_ALIASES,
    **AGS_ALIASES,
    **PUEBLA_ALIASES,
    **HIDALGO_ALIASES,
    **TAMAULIPAS_ALIASES,
    **OAXACA_ALIASES,
    **TLAXCALA_ALIASES,
    **YUCATAN_ALIASES,
    **COAHUILA_ALIASES,
    #################### Edo de Mexico ###########################
    **EDO_MX_BOROUGHS,
    #########################################################
}


AMBIG_STATE_MUNI = {
    "aguascalientes": "aguascalientes",
    "campeche": "san francisco de campeche",
    "chihuahua": "chihuahua",
    "durango": "victoria de durango",
    "guanajuato": "guanajuato",
    "oaxaca": "oaxaca de juarez",
    "puebla": "puebla",
    "san luis potosi": "san luis potosi",
    "zacatecas": "zacatecas",
    "tlaxcala": "tlaxcala de xicohtencatl",
}


# Mini-catálogo por estado (para fuzzy si no hay match directo por alias)
MUNI_BY_STATE_MINI = {
    "estado de mexico": EDO_MX_BOROUGHS,
    "nuevo leon": NUEVO_LEON_ALIASES,
    "san luis potosi": SAN_LUIS_POTOSI_ALIASES,
    "puebla": PUEBLA_ALIASES,
    "tamaulipas": TAMAULIPAS_ALIASES,
    "hidalgo": HIDALGO_ALIASES,
    "aguascalientes" : AGS_ALIASES,
    "coahuila": COAHUILA_ALIASES,
    "yucatan": YUCATAN_ALIASES,
    "chihuahua": CHIHUAHUA_ALIASES,
    "durango": DURANGO_ALIASES,
    "guerrero": GUERRERO_ALIASES,
    "oaxaca": OAXACA_ALIASES,
    "campeche": CAMPECHE_ALIASES,
    "tlaxcala": TLAXCALA_ALIASES,
    "jalisco": JALISCO_ALIASES,
    "guanajuato": GUANAJUATO_ALIASES,
    "sinaloa": SINALOA_ALIASES,
    "cdmx": CDMX_BOROUGHS, # Orden es relevante
    # agrega más según veas en tus direcciones
}

MEDICAL_DIAG_KEYWORDS = {
    "diagnostico", "diagnóstico", "tratar", "tratamiento", "recetar", "dosis"
}

URGENT = {
    "emergencia", "sangre", "sangrado", "infarto", "paro", "desmayé", "desmayo"
}



"""
Asistencia de Service Population, basado en los campos existentes en el excel. Si el servicio no está declarado,
pero se encuentra un horario en la columna, se agrega.
"""

NEG_MARKS = {"* * *","x","*","nan","na","s/d","sin dato"}

def add_services_from_staff(row, servicios):
    # keys exactas de tu hoja:
    headers = {
        "MÉDICO GENERAL": "consulta_general",
        "DENTISTA": "odontologia",
        "OPTOMETRISTA": "optometrista",
        "NUTRIOLOGO": "nutricion",
        "PSICOLOGO": "psicologia",
    }
    extra_info = []
    for hdr, srv in headers.items():
        val = str(row.get(hdr, "") or "").strip().lower()
        if val and val not in NEG_MARKS:
            if srv not in servicios:
                servicios.append(srv)
            extra_info.append(val)
    return servicios, extra_info