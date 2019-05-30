/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "raw/RawModel.hpp"

#include "FbxBlendShapesAccess.hpp"
#include "FbxLayerElementAccess.hpp"
#include "FbxSkinningAccess.hpp"
#include "materials/RoughnessMetallicMaterials.hpp"
#include "materials/TraditionalMaterials.hpp"

namespace FBX2glTF {

struct Polygon {
  Polygon(int fbxPolyIndex)
      : fbxPolyIndex(fbxPolyIndex), hasTranslucentVertices(false), rawPolyIndex(-1) {}
  const int fbxPolyIndex;
  bool hasTranslucentVertices;
  int rawPolyIndex;
};

struct Triangle {
  Triangle(std::shared_ptr<Polygon> polygon, int v1, int v2, int v3)
      : polygon(polygon), vertexIndices{v1, v2, v3}, rawVertexIndices{-1, -1, -1} {}
  const std::shared_ptr<Polygon> polygon;
  const int vertexIndices[3];
  int rawVertexIndices[3];
};

struct Triangulation {
  Triangulation(FbxMesh* mesh) : mesh(mesh) {}
  FbxMesh* const mesh;
  std::vector<const FbxBlendShapesAccess::TargetShape*> targetShapes;
  std::vector<std::shared_ptr<Polygon>> polygons;
  std::vector<std::shared_ptr<Triangle>> triangles;
};

struct FbxMeshAccess {
  FbxMesh* mesh;
  const FbxLayerElementAccess<FbxVector4> normalLayer;
  const FbxLayerElementAccess<FbxVector4> binormalLayer;
  const FbxLayerElementAccess<FbxVector4> tangentLayer;
  const FbxLayerElementAccess<FbxColor> colorLayer;
  const FbxLayerElementAccess<FbxVector2> uvLayer0;
  const FbxLayerElementAccess<FbxVector2> uvLayer1;
  const FbxSkinningAccess skinning;
  const FbxMaterialsAccess materials;
  const FbxBlendShapesAccess blendShapes;

  FbxMeshAccess(
      FbxScene* scene,
      FbxNode* node,
      const std::map<const FbxTexture*, FbxString>& textureLocations);
};

enum FbxMeshTriangulationMethod {
  FBX_SDK,
  FB_NGON_ENCODING,
};

class FbxMesh2Raw {
 public:
  FbxMesh2Raw(
      RawModel& raw,
      FbxScene* scene,
      FbxNode* node,
      FbxMeshTriangulationMethod triangulationMethod,
      const std::map<const FbxTexture*, FbxString>& textureLocations);

  void Process();

 private:
  int ProcessSurface(Triangulation& triangulation, const FbxMeshAccess& meshAccess);
  void ProcessTriangles(
      Triangulation& triangulation,
      int rawSurfaceIndex,
      const FbxMeshAccess& meshAccess);
  RawMaterialType GetMaterialType(
      const int textures[RAW_TEXTURE_USAGE_MAX],
      const bool vertexTransparency,
      const bool isSkinned);
  int GetMaterial(
      const std::shared_ptr<FbxMaterialInfo> fbxMaterial,
      const std::vector<std::string> userProperties,
      const bool hasTranslucentVertices,
      const bool isSkinned);
  void ProcessPolygons(
      Triangulation& triangulation,
      int rawSurfaceIndex,
      const FbxMeshAccess& meshAccess);
  std::unique_ptr<Triangulation> TriangulateUsingSDK();
  std::unique_ptr<Triangulation> TriangulateForNgonEncoding();

  RawModel& raw;
  FbxScene* scene;
  FbxNode* node;
  const FbxMeshTriangulationMethod triangulationMethod;
  const std::map<const FbxTexture*, FbxString>& textureLocations;
};
} // namespace FBX2glTF
