"use client";

import { MTLLoader } from "@/lib/MTLLoader";
import { OBJLoader } from "@/lib/OBJLoader";
import { useLoader } from "@react-three/fiber";

export function Avatar({
  data,
}: {
  data: {
    mtl: string;
    obj: string;
  };
}) {
  const mtl = useLoader(MTLLoader, data.mtl);
  const obj = useLoader(
    OBJLoader,
    data.obj,
    (loader) => {
      loader.setMaterials(mtl);
    },
    () => {}
  );
  return (
    <primitive
      object={obj}
      scale={2}
      position={[0, -206, -5]}
      rotation={[0, Math.PI, 0]}
    />
  );
}
