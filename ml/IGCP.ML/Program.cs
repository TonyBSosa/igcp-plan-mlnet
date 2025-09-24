using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Data.SqlClient;
using Microsoft.ML;
using Microsoft.ML.Data;

// ================== CONFIG ==================
const string CONN =
    "Server=.;Database=iAcademicGenerator_copia;Trusted_Connection=True;TrustServerCertificate=True";
const string PERIODO = "20240";  // <-- cámbialo si quieres
const bool   SOLO_INFORMATICA = true;

// Salidas
var OUT_DIR = @"ml/IGCP.ML/out";
Directory.CreateDirectory(OUT_DIR);
var OUT_FILE = Path.Combine(OUT_DIR, "pred_ofertas_resultados.csv");

// ================== MLContext ==================
var ml = new MLContext(seed: 42);

// ================== Carga desde BD ==================
var trainRowsDb = LoadTrainRows(CONN, PERIODO, SOLO_INFORMATICA);
var predRowsDb  = LoadPredictRows(CONN, PERIODO, SOLO_INFORMATICA);

// Map a feature-rows (sin nulos, numéricos en float)
var trainFe = trainRowsDb.Select(ToFeTrain).ToList();
var predFe  = predRowsDb.Select(ToFePredict).ToList();

// IDataView
var trainData = ml.Data.LoadFromEnumerable(trainFe.Where(r => r.ofe_matriculados >= 0)); // solo donde hay label
var predData  = ml.Data.LoadFromEnumerable(predFe);

// ================== Selección de columnas ==================
// Baseline planificación (sin cupo / sin doc/sec/aula)
string[] catCols = { "mod_codigo", "cam_codigo", "ofe_modalidad_programa" };
string[] numCols = {
    "ofe_modulo","ofe_semestre","ofe_anio","ofe_nivel",
    "ofe_presencialidad_obligatoria","ofe_duracion_clase","ofe_es_core",
    "ofe_dias_habiles","ofe_hora_min","pre_solicitudes"
};

// Prepro: categóricas -> OneHotHash; numéricas -> normalización
var catPairs = catCols.Select(c => new InputOutputColumnPair(c + "_oh", c)).ToArray();

var preproc = ml.Transforms.Categorical.OneHotHashEncoding(catPairs, numberOfBits: 14)
    .Append(ml.Transforms.Concatenate("Features",
        numCols.Concat(catCols.Select(c => c + "_oh")).ToArray()))
    .Append(ml.Transforms.NormalizeMinMax("Features"));

// ============== 1) REGRESIÓN: matrícula esperada ==============
var regPipeline = preproc
    .Append(ml.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(FeTrain.ofe_matriculados)))
    .Append(ml.Regression.Trainers.FastTree(numberOfLeaves: 64, minimumExampleCountPerLeaf: 10));

var regModel = regPipeline.Fit(trainData);
var regPred  = regModel.Transform(predData);
var regEnum  = ml.Data.CreateEnumerable<RegPrediction>(regPred, reuseRowObject: false).ToList();

// ============== 2) CLASIFICACIÓN: abrir / no abrir ==============
var clsPipeline = preproc
    .Append(ml.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(FeTrain.abierta)))
    .Append(ml.BinaryClassification.Trainers.FastTree());

var clsModel = clsPipeline.Fit(trainData);
var clsPred  = clsModel.Transform(predData);
var clsEnum  = ml.Data.CreateEnumerable<ClsPrediction>(clsPred, reuseRowObject: false).ToList();

// ============== 3) Escribir resultados combinados ==============
var lines = new List<string> {
    "per_codigo,mod_codigo,cam_codigo,sec_codigo,doc_codigo,matric_pred,abrir_pred,prob"
};
for (int i = 0; i < predFe.Count; i++)
{
    var x = predFe[i];
    var m = regEnum[i].Score;                         // matrícula predicha
    var a = clsEnum[i].PredictedLabel ? 1 : 0;        // abrir (0/1)
    var p = clsEnum[i].Probability;                   // probabilidad de abrir

    lines.Add($"{x.per_codigo},{x.mod_codigo},{x.cam_codigo},{x.sec_codigo},{x.doc_codigo},{m:F2},{a},{p:F3}");
}
File.WriteAllLines(OUT_FILE, lines);
Console.WriteLine($"[DONE] {OUT_FILE}");

// ================== Tipos/POCOs ==================
public class InputRowTrain
{
    public string per_codigo = "";
    public string mod_codigo = "";
    public string auc_codigo = "";
    public string cam_codigo = "";
    public string sec_codigo = "";
    public string doc_codigo = "";
    public int?   ofe_modulo;
    public int?   ofe_semestre;
    public int?   ofe_anio;
    public int?   ofe_nivel;
    public int?   ofe_cupo;
    public int?   ofe_matriculados;
    public int?   ofe_presencialidad_obligatoria;
    public DateTime? ofe_fecha_inicio;
    public DateTime? ofe_fecha_fin;
    public int?   ofe_duracion_clase;
    public string? ofe_modalidad_programa;
    public int?   ofe_es_core;
    public int?   ofe_dias_habiles;
    public int?   ofe_hora_min;
    public int?   flag_informatica;
    public int?   pre_solicitudes;
    public bool   abierta; // label derivada
}

public class InputRowPredict
{
    public string per_codigo = "";
    public string mod_codigo = "";
    public string auc_codigo = "";
    public string cam_codigo = "";
    public string sec_codigo = "";
    public string doc_codigo = "";
    public int?   ofe_modulo;
    public int?   ofe_semestre;
    public int?   ofe_anio;
    public int?   ofe_nivel;
    public int?   ofe_cupo;
    public int?   ofe_presencialidad_obligatoria;
    public DateTime? ofe_fecha_inicio;
    public DateTime? ofe_fecha_fin;
    public int?   ofe_duracion_clase;
    public string? ofe_modalidad_programa;
    public int?   ofe_es_core;
    public int?   ofe_dias_habiles;
    public int?   ofe_hora_min;
    public int?   flag_informatica;
    public int?   pre_solicitudes;
}

// Filas con tipos que ML.NET consume directamente (floats/strings no nulos)
public class FeTrain
{
    public string per_codigo = "";
    public string mod_codigo = "";
    public string cam_codigo = "";
    public string sec_codigo = "";
    public string doc_codigo = "";
    public string ofe_modalidad_programa = "";

    public float ofe_modulo;
    public float ofe_semestre;
    public float ofe_anio;
    public float ofe_nivel;
    public float ofe_presencialidad_obligatoria;
    public float ofe_duracion_clase;
    public float ofe_es_core;
    public float ofe_dias_habiles;
    public float ofe_hora_min;
    public float pre_solicitudes;

    public float ofe_matriculados; // Label regresión
    public bool  abierta;           // Label clasificación
}
public class FePredict
{
    public string per_codigo = "";
    public string mod_codigo = "";
    public string cam_codigo = "";
    public string sec_codigo = "";
    public string doc_codigo = "";
    public string ofe_modalidad_programa = "";

    public float ofe_modulo;
    public float ofe_semestre;
    public float ofe_anio;
    public float ofe_nivel;
    public float ofe_presencialidad_obligatoria;
    public float ofe_duracion_clase;
    public float ofe_es_core;
    public float ofe_dias_habiles;
    public float ofe_hora_min;
    public float pre_solicitudes;
}

public class RegPrediction { public float Score { get; set; } }
public class ClsPrediction { public bool PredictedLabel { get; set; } public float Probability { get; set; } public float Score { get; set; } }

// ================== Mapping helpers ==================
static FeTrain ToFeTrain(InputRowTrain r) => new FeTrain {
    per_codigo = r.per_codigo,
    mod_codigo = r.mod_codigo,
    cam_codigo = r.cam_codigo,
    sec_codigo = r.sec_codigo,
    doc_codigo = r.doc_codigo,
    ofe_modalidad_programa = r.ofe_modalidad_programa ?? "UNKNOWN",

    ofe_modulo   = (float)(r.ofe_modulo   ?? 0),
    ofe_semestre = (float)(r.ofe_semestre ?? 0),
    ofe_anio     = (float)(r.ofe_anio     ?? 0),
    ofe_nivel    = (float)(r.ofe_nivel    ?? 0),
    ofe_presencialidad_obligatoria = (float)(r.ofe_presencialidad_obligatoria ?? 0),
    ofe_duracion_clase = (float)(r.ofe_duracion_clase ?? 0),
    ofe_es_core  = (float)(r.ofe_es_core  ?? 0),
    ofe_dias_habiles = (float)(r.ofe_dias_habiles ?? 0),
    ofe_hora_min = (float)(r.ofe_hora_min ?? 0),
    pre_solicitudes = (float)(r.pre_solicitudes ?? 0),

    ofe_matriculados = (float)(r.ofe_matriculados ?? -1), // -1 => filtrar
    abierta = r.abierta
};
static FePredict ToFePredict(InputRowPredict r) => new FePredict {
    per_codigo = r.per_codigo,
    mod_codigo = r.mod_codigo,
    cam_codigo = r.cam_codigo,
    sec_codigo = r.sec_codigo,
    doc_codigo = r.doc_codigo,
    ofe_modalidad_programa = r.ofe_modalidad_programa ?? "UNKNOWN",

    ofe_modulo   = (float)(r.ofe_modulo   ?? 0),
    ofe_semestre = (float)(r.ofe_semestre ?? 0),
    ofe_anio     = (float)(r.ofe_anio     ?? 0),
    ofe_nivel    = (float)(r.ofe_nivel    ?? 0),
    ofe_presencialidad_obligatoria = (float)(r.ofe_presencialidad_obligatoria ?? 0),
    ofe_duracion_clase = (float)(r.ofe_duracion_clase ?? 0),
    ofe_es_core  = (float)(r.ofe_es_core  ?? 0),
    ofe_dias_habiles = (float)(r.ofe_dias_habiles ?? 0),
    ofe_hora_min = (float)(r.ofe_hora_min ?? 0),
    pre_solicitudes = (float)(r.pre_solicitudes ?? 0)
};

// ================== Lectura ADO.NET (vistas relajadas) ==================
static List<InputRowTrain> LoadTrainRows(string connStr, string? per, bool onlyInf)
{
    const string sql = @"
SELECT *
FROM ML.v_ofe_train_relaxed
WHERE (@per IS NULL OR per_codigo = @per)
  AND (@onlyInf = 0 OR flag_informatica = 1);";

    var rows = new List<InputRowTrain>();
    using var cn = new SqlConnection(connStr);
    using var cmd = new SqlCommand(sql, cn);
    cmd.Parameters.AddWithValue("@per",  (object?)per ?? DBNull.Value);
    cmd.Parameters.AddWithValue("@onlyInf", onlyInf ? 1 : 0);
    cn.Open();
    using var rd = cmd.ExecuteReader();
    while (rd.Read())
    {
        var r = new InputRowTrain {
            per_codigo = GetStr(rd,"per_codigo") ?? "",
            mod_codigo = GetStr(rd,"mod_codigo") ?? "UNKNOWN",
            auc_codigo = GetStr(rd,"auc_codigo") ?? "UNKNOWN",
            cam_codigo = GetStr(rd,"cam_codigo") ?? "UNKNOWN",
            sec_codigo = GetStr(rd,"sec_codigo") ?? "UNKNOWN",
            doc_codigo = GetStr(rd,"doc_codigo") ?? "UNKNOWN",
            ofe_modulo   = GetInt(rd,"ofe_modulo"),
            ofe_semestre = GetInt(rd,"ofe_semestre"),
            ofe_anio     = GetInt(rd,"ofe_anio"),
            ofe_nivel    = GetInt(rd,"ofe_nivel"),
            ofe_cupo     = GetInt(rd,"ofe_cupo"),
            ofe_matriculados = GetInt(rd,"ofe_matriculados"),
            ofe_presencialidad_obligatoria = GetInt(rd,"ofe_presencialidad_obligatoria"),
            ofe_fecha_inicio = GetDate(rd,"ofe_fecha_inicio"),
            ofe_fecha_fin    = GetDate(rd,"ofe_fecha_fin"),
            ofe_duracion_clase = GetInt(rd,"ofe_duracion_clase"),
            ofe_modalidad_programa = GetStr(rd,"ofe_modalidad_programa"),
            ofe_es_core  = GetInt(rd,"ofe_es_core"),
            ofe_dias_habiles = GetInt(rd,"ofe_dias_habiles"),
            ofe_hora_min = GetInt(rd,"ofe_hora_min"),
            flag_informatica = GetInt(rd,"flag_informatica"),
            pre_solicitudes  = GetInt(rd,"pre_solicitudes"),
            abierta = GetBool(rd,"abierta_label_rule")
        };
        rows.Add(r);
    }
    return rows;
}
static List<InputRowPredict> LoadPredictRows(string connStr, string? per, bool onlyInf)
{
    const string sql = @"
SELECT *
FROM ML.v_ofe_predict_relaxed
WHERE (@per IS NULL OR per_codigo = @per)
  AND (@onlyInf = 0 OR flag_informatica = 1);";

    var rows = new List<InputRowPredict>();
    using var cn = new SqlConnection(connStr);
    using var cmd = new SqlCommand(sql, cn);
    cmd.Parameters.AddWithValue("@per",  (object?)per ?? DBNull.Value);
    cmd.Parameters.AddWithValue("@onlyInf", onlyInf ? 1 : 0);
    cn.Open();
    using var rd = cmd.ExecuteReader();
    while (rd.Read())
    {
        var r = new InputRowPredict {
            per_codigo = GetStr(rd,"per_codigo") ?? "",
            mod_codigo = GetStr(rd,"mod_codigo") ?? "UNKNOWN",
            auc_codigo = GetStr(rd,"auc_codigo") ?? "UNKNOWN",
            cam_codigo = GetStr(rd,"cam_codigo") ?? "UNKNOWN",
            sec_codigo = GetStr(rd,"sec_codigo") ?? "UNKNOWN",
            doc_codigo = GetStr(rd,"doc_codigo") ?? "UNKNOWN",
            ofe_modulo   = GetInt(rd,"ofe_modulo"),
            ofe_semestre = GetInt(rd,"ofe_semestre"),
            ofe_anio     = GetInt(rd,"ofe_anio"),
            ofe_nivel    = GetInt(rd,"ofe_nivel"),
            ofe_cupo     = GetInt(rd,"ofe_cupo"),
            ofe_presencialidad_obligatoria = GetInt(rd,"ofe_presencialidad_obligatoria"),
            ofe_fecha_inicio = GetDate(rd,"ofe_fecha_inicio"),
            ofe_fecha_fin    = GetDate(rd,"ofe_fecha_fin"),
            ofe_duracion_clase = GetInt(rd,"ofe_duracion_clase"),
            ofe_modalidad_programa = GetStr(rd,"ofe_modalidad_programa"),
            ofe_es_core  = GetInt(rd,"ofe_es_core"),
            ofe_dias_habiles = GetInt(rd,"ofe_dias_habiles"),
            ofe_hora_min = GetInt(rd,"ofe_hora_min"),
            flag_informatica = GetInt(rd,"flag_informatica"),
            pre_solicitudes  = GetInt(rd,"pre_solicitudes")
        };
        rows.Add(r);
    }
    return rows;
}

// Helpers ADO
static string?   GetStr (SqlDataReader r, string col) => r.IsDBNull(r.GetOrdinal(col)) ? null : r.GetString(r.GetOrdinal(col));
static int?      GetInt (SqlDataReader r, string col) => r.IsDBNull(r.GetOrdinal(col)) ? (int?)null : r.GetInt32(r.GetOrdinal(col));
static DateTime? GetDate(SqlDataReader r, string col) => r.IsDBNull(r.GetOrdinal(col)) ? (DateTime?)null : r.GetDateTime(r.GetOrdinal(col));
static bool      GetBool(SqlDataReader r, string col)
{
    if (r.IsDBNull(r.GetOrdinal(col))) return false;
    var t = r.GetFieldType(r.GetOrdinal(col));
    if (t == typeof(bool)) return r.GetBoolean(r.GetOrdinal(col));
    if (t == typeof(int))  return r.GetInt32(r.GetOrdinal(col)) != 0;
    return false;
}
