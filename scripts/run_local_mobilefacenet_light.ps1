$ErrorActionPreference = "Stop"

$Python = "D:\Anaconda3\envs\ml-gpu\python.exe"
$ProjectRoot = "F:\Internship\Bytedance\face-ai-project"

Set-Location $ProjectRoot

# Local thermal-friendly baseline:
# - lightweight MobileFaceNet backbone
# - smaller identity/image subset
# - batch size 24 instead of 64
# - no DataLoader workers to avoid high CPU load
# - cooldown between epochs to reduce sustained heat
$TrainArgs = @(
    "-u", "src\recognition\train_arcface_celeba_subset.py",
    "--dataset-format", "folder",
    "--data-root", "data\processed\ms1m-aligned-112-5000",
    "--dataset-name", "MS1M-local-mobilefacenet-light",
    "--backbone", "mobilefacenet",
    "--pretrained", "none",
    "--embedding-dim", "256",
    "--num-identities", "2000",
    "--min-train-images", "5",
    "--max-train-images-per-identity", "12",
    "--max-val-images-per-identity", "3",
    "--epochs", "12",
    "--batch-size", "24",
    "--sampler", "random",
    "--scheduler", "none",
    "--lr", "0.0005",
    "--weight-decay", "0.0001",
    "--device", "cuda",
    "--num-workers", "0",
    "--epoch-cooldown-seconds", "60",
    "--output", "models\checkpoints\mobilefacenet_ms1m_local_light_best.pt",
    "--report", "outputs\reports\mobilefacenet_ms1m_local_light_result.txt",
    "--history-json", "outputs\reports\mobilefacenet_ms1m_local_light_history.json",
    "--curve", "outputs\images\mobilefacenet_ms1m_local_light_curves.jpg"
)

& $Python @TrainArgs | Tee-Object -FilePath "outputs\reports\mobilefacenet_ms1m_local_light_console_log.txt"

& $Python "src\recognition\evaluate_lfw_10fold_resnet_arcface.py" `
    --checkpoint "models\checkpoints\mobilefacenet_ms1m_local_light_best.pt" `
    --batch-size 128 `
    --device cuda `
    --report "outputs\reports\mobilefacenet_ms1m_local_light_lfw_10fold_result.txt" `
    --json-report "outputs\reports\mobilefacenet_ms1m_local_light_lfw_10fold_result.json"
