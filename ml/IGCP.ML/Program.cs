using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Data.SqlClient;               // proveedor SQL estable
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.Extensions.Configuration;

// =================== Config y cadena de conexión ===================
var cfg = new ConfigurationBuilder()
    .AddUserSecrets<Program>()  // lee "ConnStr" guardado con user-secrets
    .Build();

// 3 fuentes: UserSecrets -> ConnectionStrings -> variable de entorno
string? conn =
    cfg["ConnStr"]                                   // 1) user-secrets
    ?? cfg.GetConnectionString("DefaultConnection")  // 2) appsettings (opcional)
    ?? Environment.GetEnvironmentVariable("CONNSTR");// 3) env var (opcional)

if (string.IsNullOrWhiteSpace(conn))
{
    Console.Error.WriteLine("ERROR: No se encontró cadena de conexión.");
    Console.Error.WriteLine("Ejemplo:");
    Console.Error.WriteLine("  dotnet user-secrets set \"ConnStr\" \"Server=iAcademicGenerator.mssql.somee.com,1433;Database=iAcademicGenerator;User ID=oscarvaldivieso_SQLLogin_1;Password=admin123;Encrypt=True;TrustServerCertificate=True;Connection Timeout=60;\"");
    return;
}
Console.WriteLine($"[INFO] ConnStr OK (len={conn.Length})");

// =================== Smoke test de conexión ===================
try
{
    using var test = new SqlConnection(conn);
    test.Open(); // síncrono
    Console.WriteLine("[INFO] Conexión a SQL OK");
}
catch (SqlException ex)
{
    Console.Error.WriteLine($"[SQL ERROR {ex.Number}] {ex.Message}");
    return;
}

// =================== Consulta de datos (vista -> tabla) ===================
const string Q_VIEW  = "SELECT car_codigo, mat_codigo, semestre, modulo FROM ML.vw_features_plan";
const string Q_TABLE = "SELECT car_codigo, mat_codigo, semestre, modulo FROM UNI.car_plan_materia";

var rows = new List<Row>();

using (var cn = new SqlConnection(conn))
{
    cn.Open();

    void Read(string sql)
    {
        using var cmd = new SqlCommand(sql, cn);
        using var rd  = cmd.ExecuteReader();
        while (rd.Read())
        {
            rows.Add(new Row
            {
                car_codigo = rd.GetString(0),
                mat_codigo = rd.GetString(1),
                semestre   = rd.IsDBNull(2) ? 0 : rd.GetInt32(2),
                modulo     = rd.IsDBNull(3) ? "" : rd.GetString(3)   // nunca null
            });
        }
    }

    try
    {
        Read(Q_VIEW);
        Console.WriteLine("[INFO] Fuente: ML.vw_features_plan");
    }
    catch (SqlException ex) when (ex.Number == 208)
    {
        Console.WriteLine("[WARN] Vista no existe. Usando UNI.car_plan_materia.");
        Read(Q_TABLE);
    }
}

Console.WriteLine($"[INFO] Loaded {rows.Count} rows");

// =================== Preprocesado ML.NET ===================
var ml = new MLContext(seed: 42);
var data = ml.Data.LoadFromEnumerable(rows);

var pipeline =
    ml.Transforms.Categorical.OneHotEncoding(new[]
    {
        new InputOutputColumnPair("car_ohe","car_codigo"),
        new InputOutputColumnPair("mat_ohe","mat_codigo"),
        new InputOutputColumnPair("mod_ohe","modulo")
    })
    .Append(ml.Transforms.Conversion.ConvertType("sem_f", "semestre", DataKind.Single)) // Int32 → Single
    .Append(ml.Transforms.Concatenate("Features","car_ohe","mat_ohe","mod_ohe","sem_f"));

var pre = pipeline.Fit(data);

// =================== Artefactos ===================
Directory.CreateDirectory("out");
ml.Model.Save(pre, data.Schema, "out/preprocess_plan.zip");

var lines = new[] { "car_codigo,mat_codigo,semestre,modulo" }
    .Concat(rows.Select(r => $"{r.car_codigo},{r.mat_codigo},{r.semestre},{r.modulo}"));
File.WriteAllLines("out/plan_snapshot.csv", lines);

Console.WriteLine("[DONE] Saved out/preprocess_plan.zip and out/plan_snapshot.csv");

// =================== POCO ===================
public class Row
{
    public string car_codigo { get; set; } = "";
    public string mat_codigo { get; set; } = "";
    public int    semestre   { get; set; }
    public string modulo     { get; set; } = "";
}
