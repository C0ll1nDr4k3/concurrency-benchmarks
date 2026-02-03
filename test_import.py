import nilvec
print("Imported nilvec:", nilvec)
import nilvec.benchmark
print("Imported nilvec.benchmark")
try:
    idx = nilvec.HNSWVanilla(128)
    print("Created Index successfully")
except Exception as e:
    print(f"Failed to create index: {e}")
