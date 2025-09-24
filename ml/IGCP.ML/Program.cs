using System.Data;
using Microsoft.Data.SqlClient;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

const string CONN =
    "Server=.;Database=iAcademicGenerator_copia;Trusted_Connection=True;TrustServerCertificate=True";

// Deja null para traer todos los periodos
const string? PERIODO = "20240";

// Umbral simple para decidir abrir por regresión (puedes calibrarlo)
const float OPEN_THRESHOLD = 12f;   // si matric_pred >= 12 => abrir
const float SIGMA = 5f;             // suavizado para prob (logística)

var OUT_DIR  = Path.Combine(AppContext.BaseDirectory, "out");
Directory.CreateDirectory(OUT_DIR);
var OUT_FILE = Path.Combine(OUT_DIR, "pred_ofertas_resultados.csv");

var ml = new MLContext(seed: 42);

// ==================== 1) Cargar datos desde SQL =====================
List<InputRowTrain> trainRows = LoadTrain(CONN, PERIODO);
List<InputRowPredict> predRows = LoadPredict(CONN, PERIODO);

if (trainRows.Count == 0 || predRows.Count == 0)
{
    Console.WriteLine($"[WARN] train={trainRows.Count}, predict={predRows.Count}. Revisa las vistas o el periodo.");
}

List<FeTrain>   trainFe = trainRows.Select(ToFeTrain).ToList();
List<FePredict> predFe  = predRows .Select(ToFePredict).ToList();

IDataView trainData = ml.Data.LoadFromEnumerable(trainFe);
IDataView predData  = ml.Data.LoadFromEnumerable(predFe);

// ==================== 2) Preprocesado (solo features) ===============
string[] catCols = new[] { "per_codigo", "mod_codigo", "cam_codigo", "sec_codigo", "doc_codigo", "ofe_modalidad_programa" };
string[] numCols = new[] { "ofe_modulo", "ofe_semestre", "ofe_anio", "ofe_nivel",
                           "ofe_presencialidad_obligatoria", "ofe_duracion_clase", "ofe_es_core",
                           "ofe_dias_habiles", "ofe_hora_min", "pre_solicitudes" };

// pares entrada/salida para OneHotHashEncoding
var catPairs = catCols
    .Select(c => new InputOutputColumnPair(outputColumnName: c, inputColumnName: c))
    .ToArray();

var preproc =
    ml.Transforms.Categorical.OneHotHashEncoding(
            columns: catPairs,
            numberOfBits: 15,
            outputKind: OneHotEncodingEstimator.OutputKind.Indicator)
      .Append(ml.Transforms.Concatenate("Features", catCols.Concat(numCols).ToArray()))
      .Append(ml.Transforms.NormalizeMinMax("Features"));

// El preprocesador se ajusta con TRAIN y se reutiliza
var preTf = preproc.Fit(trainData);

// ==================== 3) REGRESIÓN (matrícula) ======================
// Label SOLO en TRAIN
var labelerReg   = ml.Transforms.CopyColumns("Label", nameof(FeTrain.ofe_matriculados));
var labelTfReg   = labelerReg.Fit(trainData);
var trainLblReg  = labelTfReg.Transform(trainData);
var trainPrepReg = preTf.Transform(trainLblReg);

// Entrenar
var regTrainer = ml.Regression.Trainers.FastTree(numberOfLeaves: 64, minimumExampleCountPerLeaf: 10);
var regModel   = regTrainer.Fit(trainPrepReg);

// Predecir (NO copiar label aquí)
var predPrepReg = preTf.Transform(predData);
var regPred     = regModel.Transform(predPrepReg);
var regEnum     = ml.Data.CreateEnumerable<RegPrediction>(regPred, reuseRowObject: false).ToList();

// ==================== 4) Exportar resultados combinados =============
var lines = new List<string> {
    "per_codigo,mod_codigo,cam_codigo,sec_codigo,doc_codigo,matric_pred,abrir_pred,prob"
};

int n = Math.Min(predFe.Count, regEnum.Count);
for (int i = 0; i < n; i++)
{
    var x = predFe[i];
    var m = regEnum[i].Score;                  // matrícula predicha

    // derivar abrir y probabilidad sin columna 'abierta'
    var a = (m >= OPEN_THRESHOLD) ? 1 : 0;
    var p = Logistic((m - OPEN_THRESHOLD) / SIGMA); // 0..1

    lines.Add($"{x.per_codigo},{x.mod_codigo},{x.cam_codigo},{x.sec_codigo},{x.doc_codigo},{m:F2},{a},{p:F3}");
}

File.WriteAllLines(OUT_FILE, lines);
Console.WriteLine($"[DONE] {OUT_FILE}  (rows={n})");


// ====================================================================
// =========================== Helpers ================================
// ====================================================================

static float Logistic(float z) => 1f / (1f + (float)Math.Exp(-z));

static List<InputRowTrain> LoadTrain(string conn, string? per)
{
    var list = new List<InputRowTrain>();
    using var cn = new SqlConnection(conn);
    cn.Open();
    using var cmd = cn.CreateCommand();
    cmd.CommandText = @"
SELECT
  per_codigo, mod_codigo, cam_codigo, sec_codigo, doc_codigo,
  ofe_modulo, ofe_semestre, ofe_anio, ofe_nivel,
  ofe_presencialidad_obligatoria, ofe_duracion_clase, ofe_es_core,
  ofe_dias_habiles, ofe_hora_min, pre_solicitudes,
  ofe_modalidad_programa,
  ofe_matriculados       -- << SOLO esta label para regresión
FROM ML.v_ofe_train_relaxed
WHERE (@p IS NULL OR per_codigo = @p);";
    cmd.Parameters.Add(new SqlParameter("@p", SqlDbType.VarChar, 20) { Value = (object?)per ?? DBNull.Value });

    using var rd = cmd.ExecuteReader();
    while (rd.Read())
    {
        list.Add(new InputRowTrain
        {
            per_codigo = rd["per_codigo"]?.ToString() ?? "",
            mod_codigo = rd["mod_codigo"]?.ToString() ?? "",
            cam_codigo = rd["cam_codigo"]?.ToString() ?? "",
            sec_codigo = rd["sec_codigo"]?.ToString() ?? "",
            doc_codigo = rd["doc_codigo"]?.ToString() ?? "",
            ofe_modulo = rd["ofe_modulo"] as int? ?? ParseInt(rd["ofe_modulo"]),
            ofe_semestre = rd["ofe_semestre"] as int? ?? ParseInt(rd["ofe_semestre"]),
            ofe_anio = rd["ofe_anio"] as int? ?? ParseInt(rd["ofe_anio"]),
            ofe_nivel = rd["ofe_nivel"] as int? ?? ParseInt(rd["ofe_nivel"]),
            ofe_presencialidad_obligatoria = rd["ofe_presencialidad_obligatoria"] as int? ?? ParseInt(rd["ofe_presencialidad_obligatoria"]),
            ofe_duracion_clase = rd["ofe_duracion_clase"] as int? ?? ParseInt(rd["ofe_duracion_clase"]),
            ofe_es_core = rd["ofe_es_core"] as int? ?? ParseInt(rd["ofe_es_core"]),
            ofe_dias_habiles = rd["ofe_dias_habiles"] as int? ?? ParseInt(rd["ofe_dias_habiles"]),
            ofe_hora_min = rd["ofe_hora_min"] as int? ?? ParseInt(rd["ofe_hora_min"]),
            pre_solicitudes = rd["pre_solicitudes"] as int? ?? ParseInt(rd["pre_solicitudes"]),
            ofe_modalidad_programa = rd["ofe_modalidad_programa"]?.ToString(),
            ofe_matriculados = rd["ofe_matriculados"] as int? ?? ParseInt(rd["ofe_matriculados"])
        });
    }
    return list;
}

static List<InputRowPredict> LoadPredict(string conn, string? per)
{
    var list = new List<InputRowPredict>();
    using var cn = new SqlConnection(conn);
    cn.Open();
    using var cmd = cn.CreateCommand();
    cmd.CommandText = @"
SELECT
  per_codigo, mod_codigo, cam_codigo, sec_codigo, doc_codigo,
  ofe_modulo, ofe_semestre, ofe_anio, ofe_nivel,
  ofe_presencialidad_obligatoria, ofe_duracion_clase, ofe_es_core,
  ofe_dias_habiles, ofe_hora_min, pre_solicitudes,
  ofe_modalidad_programa
FROM ML.v_ofe_predict_relaxed
WHERE (@p IS NULL OR per_codigo = @p);";
    cmd.Parameters.Add(new SqlParameter("@p", SqlDbType.VarChar, 20) { Value = (object?)per ?? DBNull.Value });

    using var rd = cmd.ExecuteReader();
    while (rd.Read())
    {
        list.Add(new InputRowPredict
        {
            per_codigo = rd["per_codigo"]?.ToString() ?? "",
            mod_codigo = rd["mod_codigo"]?.ToString() ?? "",
            cam_codigo = rd["cam_codigo"]?.ToString() ?? "",
            sec_codigo = rd["sec_codigo"]?.ToString() ?? "",
            doc_codigo = rd["doc_codigo"]?.ToString() ?? "",
            ofe_modulo = rd["ofe_modulo"] as int? ?? ParseInt(rd["ofe_modulo"]),
            ofe_semestre = rd["ofe_semestre"] as int? ?? ParseInt(rd["ofe_semestre"]),
            ofe_anio = rd["ofe_anio"] as int? ?? ParseInt(rd["ofe_anio"]),
            ofe_nivel = rd["ofe_nivel"] as int? ?? ParseInt(rd["ofe_nivel"]),
            ofe_presencialidad_obligatoria = rd["ofe_presencialidad_obligatoria"] as int? ?? ParseInt(rd["ofe_presencialidad_obligatoria"]),
            ofe_duracion_clase = rd["ofe_duracion_clase"] as int? ?? ParseInt(rd["ofe_duracion_clase"]),
            ofe_es_core = rd["ofe_es_core"] as int? ?? ParseInt(rd["ofe_es_core"]),
            ofe_dias_habiles = rd["ofe_dias_habiles"] as int? ?? ParseInt(rd["ofe_dias_habiles"]),
            ofe_hora_min = rd["ofe_hora_min"] as int? ?? ParseInt(rd["ofe_hora_min"]),
            pre_solicitudes = rd["pre_solicitudes"] as int? ?? ParseInt(rd["pre_solicitudes"]),
            ofe_modalidad_programa = rd["ofe_modalidad_programa"]?.ToString()
        });
    }
    return list;
}

static int ParseInt(object? o)
{
    if (o == null || o == DBNull.Value) return 0;
    if (int.TryParse(o.ToString(), out var v)) return v;
    return 0;
}

// ========================== Mapeos ==============================
static FeTrain ToFeTrain(InputRowTrain r) => new FeTrain
{
    per_codigo = r.per_codigo,
    mod_codigo = r.mod_codigo,
    cam_codigo = r.cam_codigo,
    sec_codigo = r.sec_codigo,
    doc_codigo = r.doc_codigo,
    ofe_modalidad_programa = r.ofe_modalidad_programa ?? "UNKNOWN",

    ofe_modulo   = r.ofe_modulo,
    ofe_semestre = r.ofe_semestre,
    ofe_anio     = r.ofe_anio,
    ofe_nivel    = r.ofe_nivel,
    ofe_presencialidad_obligatoria = r.ofe_presencialidad_obligatoria,
    ofe_duracion_clase = r.ofe_duracion_clase,
    ofe_es_core  = r.ofe_es_core,
    ofe_dias_habiles = r.ofe_dias_habiles,
    ofe_hora_min = r.ofe_hora_min,
    pre_solicitudes = r.pre_solicitudes,

    ofe_matriculados = r.ofe_matriculados
};

static FePredict ToFePredict(InputRowPredict r) => new FePredict
{
    per_codigo = r.per_codigo,
    mod_codigo = r.mod_codigo,
    cam_codigo = r.cam_codigo,
    sec_codigo = r.sec_codigo,
    doc_codigo = r.doc_codigo,
    ofe_modalidad_programa = r.ofe_modalidad_programa ?? "UNKNOWN",

    ofe_modulo   = r.ofe_modulo,
    ofe_semestre = r.ofe_semestre,
    ofe_anio     = r.ofe_anio,
    ofe_nivel    = r.ofe_nivel,
    ofe_presencialidad_obligatoria = r.ofe_presencialidad_obligatoria,
    ofe_duracion_clase = r.ofe_duracion_clase,
    ofe_es_core  = r.ofe_es_core,
    ofe_dias_habiles = r.ofe_dias_habiles,
    ofe_hora_min = r.ofe_hora_min,
    pre_solicitudes = r.pre_solicitudes
};

// ========================= POCOs ================================
// Inputs crudos desde SQL
public class InputRowTrain
{
    public string per_codigo { get; set; } = "";
    public string mod_codigo { get; set; } = "";
    public string cam_codigo { get; set; } = "";
    public string sec_codigo { get; set; } = "";
    public string doc_codigo { get; set; } = "";
    public int ofe_modulo { get; set; }
    public int ofe_semestre { get; set; }
    public int ofe_anio { get; set; }
    public int ofe_nivel { get; set; }
    public int ofe_presencialidad_obligatoria { get; set; }
    public int ofe_duracion_clase { get; set; }
    public int ofe_es_core { get; set; }
    public int ofe_dias_habiles { get; set; }
    public int ofe_hora_min { get; set; }
    public int pre_solicitudes { get; set; }
    public string? ofe_modalidad_programa { get; set; }

    public int ofe_matriculados { get; set; } // label regresión
}

public class InputRowPredict
{
    public string per_codigo { get; set; } = "";
    public string mod_codigo { get; set; } = "";
    public string cam_codigo { get; set; } = "";
    public string sec_codigo { get; set; } = "";
    public string doc_codigo { get; set; } = "";
    public int ofe_modulo { get; set; }
    public int ofe_semestre { get; set; }
    public int ofe_anio { get; set; }
    public int ofe_nivel { get; set; }
    public int ofe_presencialidad_obligatoria { get; set; }
    public int ofe_duracion_clase { get; set; }
    public int ofe_es_core { get; set; }
    public int ofe_dias_habiles { get; set; }
    public int ofe_hora_min { get; set; }
    public int pre_solicitudes { get; set; }
    public string? ofe_modalidad_programa { get; set; }
}

// Features para ML
public class FeBase
{
    public string per_codigo { get; set; } = "";
    public string mod_codigo { get; set; } = "";
    public string cam_codigo { get; set; } = "";
    public string sec_codigo { get; set; } = "";
    public string doc_codigo { get; set; } = "";
    public string ofe_modalidad_programa { get; set; } = "UNKNOWN";

    public float ofe_modulo { get; set; }
    public float ofe_semestre { get; set; }
    public float ofe_anio { get; set; }
    public float ofe_nivel { get; set; }
    public float ofe_presencialidad_obligatoria { get; set; }
    public float ofe_duracion_clase { get; set; }
    public float ofe_es_core { get; set; }
    public float ofe_dias_habiles { get; set; }
    public float ofe_hora_min { get; set; }
    public float pre_solicitudes { get; set; }
}

public class FeTrain : FeBase
{
    public float ofe_matriculados { get; set; } // Label para regresión
}

public class FePredict : FeBase { }

// Predicciones
public class RegPrediction
{
    public float Score { get; set; } // matrícula predicha
}
