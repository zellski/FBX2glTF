/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "FbxMesh2Raw.hpp"

#include <algorithm>

#include "FBX2glTF.h"

#include "raw/RawModel.hpp"

extern float scaleFactor;

using namespace FBX2glTF;

FbxMeshAccess::FbxMeshAccess(
    FbxScene* scene,
    FbxNode* node,
    const std::map<const FbxTexture*, FbxString>& textureLocations)
    : mesh(node->GetMesh()),
      normalLayer(mesh->GetElementNormal(), mesh->GetElementNormalCount()),
      binormalLayer(mesh->GetElementBinormal(), mesh->GetElementBinormalCount()),
      tangentLayer(mesh->GetElementTangent(), mesh->GetElementTangentCount()),
      colorLayer(mesh->GetElementVertexColor(), mesh->GetElementVertexColorCount()),
      uvLayer0(mesh->GetElementUV(0), mesh->GetElementUVCount()),
      uvLayer1(mesh->GetElementUV(1), mesh->GetElementUVCount()),
      skinning(mesh, scene),
      materials(mesh, textureLocations),
      blendShapes(mesh) {}

FbxMesh2Raw::FbxMesh2Raw(
    RawModel& raw,
    FbxScene* scene,
    FbxNode* node,
    FbxMeshTriangulationMethod triangulationMethod,
    const std::map<const FbxTexture*, FbxString>& textureLocations)
    : raw(raw),
      scene(scene),
      node(node),
      triangulationMethod(triangulationMethod),
      textureLocations(textureLocations) {}

void FbxMesh2Raw::Process() {
  std::unique_ptr<Triangulation> triangulation;
  switch (triangulationMethod) {
    case FBX_SDK:
      triangulation = TriangulateUsingSDK();
      break;
    case FB_NGON_ENCODING:
      triangulation = TriangulateForNgonEncoding();
      break;
  }

  const auto meshAccess = std::make_unique<FbxMeshAccess>(scene, node, textureLocations);
  int rawSurfaceIndex = ProcessSurface(*triangulation, *meshAccess);
  fmt::printf("After ProcessSurface(), rawSurfaceIndex = %d\n", rawSurfaceIndex);
  ProcessTriangles(*triangulation, rawSurfaceIndex, *meshAccess);
  ProcessPolygons(*triangulation, rawSurfaceIndex, *meshAccess);
}

int FbxMesh2Raw::ProcessSurface(Triangulation& triangulation, const FbxMeshAccess& meshAccess) {
  const long fbxSurfaceId = triangulation.mesh->GetUniqueID();

  // Associate the node to this surface
  int nodeId = raw.GetNodeById(node->GetUniqueID());
  if (nodeId >= 0) {
    RawNode& node = raw.GetNode(nodeId);
    node.surfaceId = fbxSurfaceId;
  }

  int rawSurfaceIndex = raw.GetSurfaceById(fbxSurfaceId);
  if (rawSurfaceIndex >= 0) {
    // This surface is already loaded
    return rawSurfaceIndex;
  }

  const char* meshName =
      (node->GetName()[0] != '\0') ? node->GetName() : triangulation.mesh->GetName();
  rawSurfaceIndex = raw.AddSurface(meshName, fbxSurfaceId);
  RawSurface& rawSurface = raw.GetSurface(rawSurfaceIndex);

  if (verboseOutput) {
    fmt::printf(
        "mesh %d: %s (skinned: %s)\n",
        rawSurfaceIndex,
        meshName,
        meshAccess.skinning.IsSkinned()
            ? raw.GetNode(raw.GetNodeById(meshAccess.skinning.GetRootNode())).name.c_str()
            : "NO");
  }

  raw.AddVertexAttribute(RAW_VERTEX_ATTRIBUTE_POSITION);
  if (meshAccess.normalLayer.LayerPresent()) {
    raw.AddVertexAttribute(RAW_VERTEX_ATTRIBUTE_NORMAL);
  }
  if (meshAccess.tangentLayer.LayerPresent()) {
    raw.AddVertexAttribute(RAW_VERTEX_ATTRIBUTE_TANGENT);
  }
  if (meshAccess.binormalLayer.LayerPresent()) {
    raw.AddVertexAttribute(RAW_VERTEX_ATTRIBUTE_BINORMAL);
  }
  if (meshAccess.colorLayer.LayerPresent()) {
    raw.AddVertexAttribute(RAW_VERTEX_ATTRIBUTE_COLOR);
  }
  if (meshAccess.uvLayer0.LayerPresent()) {
    raw.AddVertexAttribute(RAW_VERTEX_ATTRIBUTE_UV0);
  }
  if (meshAccess.uvLayer1.LayerPresent()) {
    raw.AddVertexAttribute(RAW_VERTEX_ATTRIBUTE_UV1);
  }
  if (meshAccess.skinning.IsSkinned()) {
    raw.AddVertexAttribute(RAW_VERTEX_ATTRIBUTE_JOINT_WEIGHTS);
    raw.AddVertexAttribute(RAW_VERTEX_ATTRIBUTE_JOINT_INDICES);
  }

  Mat4f scaleMatrix = Mat4f::FromScaleVector(Vec3f(scaleFactor, scaleFactor, scaleFactor));
  Mat4f invScaleMatrix = scaleMatrix.Inverse();

  rawSurface.skeletonRootId =
      (meshAccess.skinning.IsSkinned()) ? meshAccess.skinning.GetRootNode() : node->GetUniqueID();
  for (int jointIndex = 0; jointIndex < meshAccess.skinning.GetNodeCount(); jointIndex++) {
    const long jointId = meshAccess.skinning.GetJointId(jointIndex);
    raw.GetNode(raw.GetNodeById(jointId)).isJoint = true;

    rawSurface.jointIds.emplace_back(jointId);
    rawSurface.inverseBindMatrices.push_back(
        invScaleMatrix * toMat4f(meshAccess.skinning.GetInverseBindMatrix(jointIndex)) *
        scaleMatrix);
    rawSurface.jointGeometryMins.emplace_back(FLT_MAX, FLT_MAX, FLT_MAX);
    rawSurface.jointGeometryMaxs.emplace_back(-FLT_MAX, -FLT_MAX, -FLT_MAX);
  }

  rawSurface.blendChannels.clear();
  for (size_t channelIx = 0; channelIx < meshAccess.blendShapes.GetChannelCount(); channelIx++) {
    for (size_t targetIx = 0; targetIx < meshAccess.blendShapes.GetTargetShapeCount(channelIx);
         targetIx++) {
      const FbxBlendShapesAccess::TargetShape& shape =
          meshAccess.blendShapes.GetTargetShape(channelIx, targetIx);
      triangulation.targetShapes.push_back(&shape);
      auto& blendChannel = meshAccess.blendShapes.GetBlendChannel(channelIx);

      rawSurface.blendChannels.push_back(
          RawBlendChannel{static_cast<float>(blendChannel.deformPercent),
                          shape.normals.LayerPresent(),
                          shape.tangents.LayerPresent(),
                          blendChannel.name});
    }
  }
  return rawSurfaceIndex;
}

void FbxMesh2Raw::ProcessTriangles(
    Triangulation& triangulation,
    const int rawSurfaceIndex,
    const FbxMeshAccess& meshAccess) {
  // The FbxNode geometric transformation describes how a FbxNodeAttribute is offset from
  // the FbxNode's local frame of reference. These geometric transforms are applied to the
  // FbxNodeAttribute after the FbxNode's local transforms are computed, and are not
  // inherited across the node hierarchy.
  // Apply the geometric transform to the mesh geometry (vertices, normal etc.) because
  // glTF does not have an equivalent to the geometric transform.
  const FbxVector4 meshTranslation = node->GetGeometricTranslation(FbxNode::eSourcePivot);
  const FbxVector4 meshRotation = node->GetGeometricRotation(FbxNode::eSourcePivot);
  const FbxVector4 meshScaling = node->GetGeometricScaling(FbxNode::eSourcePivot);
  const FbxAMatrix meshTransform(meshTranslation, meshRotation, meshScaling);
  const FbxMatrix transform = meshTransform;

  // Remove translation & scaling from transforms that will be applied to normals, tangents &
  // binormals
  const FbxMatrix normalTransform(FbxVector4(), meshRotation, meshScaling);
  const FbxMatrix inverseTransposeTransform = normalTransform.Inverse().Transpose();

  const FbxVector4* controlPoints = meshAccess.mesh->GetControlPoints();
  int* allPolygonVertices = meshAccess.mesh->GetPolygonVertices();

  fmt::printf(
      "Looping over %d triangles in ProcessTriangles()...\n", triangulation.triangles.size());
  for (const auto& tri : triangulation.triangles) {
    int originalPolyIx = tri->polygon->fbxPolyIndex;

    RawVertex rawVertices[3];
    for (int vertexIndex = 0; vertexIndex < 3; vertexIndex++) {
      const int polygonVertexIndex = tri->vertexIndices[vertexIndex];
      const int controlPointIndex = allPolygonVertices[polygonVertexIndex];

      // Note that the default values here must be the same as the RawVertex default values!
      const FbxVector4 fbxPosition = transform.MultNormalize(controlPoints[controlPointIndex]);
      const FbxVector4 fbxNormal = meshAccess.normalLayer.GetElement(
          originalPolyIx,
          polygonVertexIndex,
          controlPointIndex,
          FbxVector4(0.0f, 0.0f, 0.0f, 0.0f),
          inverseTransposeTransform,
          true);
      const FbxVector4 fbxTangent = meshAccess.tangentLayer.GetElement(
          originalPolyIx,
          polygonVertexIndex,
          controlPointIndex,
          FbxVector4(0.0f, 0.0f, 0.0f, 0.0f),
          inverseTransposeTransform,
          true);
      const FbxVector4 fbxBinormal = meshAccess.binormalLayer.GetElement(
          originalPolyIx,
          polygonVertexIndex,
          controlPointIndex,
          FbxVector4(0.0f, 0.0f, 0.0f, 0.0f),
          inverseTransposeTransform,
          true);
      const FbxColor fbxColor = meshAccess.colorLayer.GetElement(
          originalPolyIx, polygonVertexIndex, controlPointIndex, FbxColor(0.0f, 0.0f, 0.0f, 0.0f));
      const FbxVector2 fbxUV0 = meshAccess.uvLayer0.GetElement(
          originalPolyIx, polygonVertexIndex, controlPointIndex, FbxVector2(0.0f, 0.0f));
      const FbxVector2 fbxUV1 = meshAccess.uvLayer1.GetElement(
          originalPolyIx, polygonVertexIndex, controlPointIndex, FbxVector2(0.0f, 0.0f));

      RawVertex& vertex = rawVertices[vertexIndex];
      vertex.position[0] = (float)fbxPosition[0] * scaleFactor;
      vertex.position[1] = (float)fbxPosition[1] * scaleFactor;
      vertex.position[2] = (float)fbxPosition[2] * scaleFactor;
      vertex.normal[0] = (float)fbxNormal[0];
      vertex.normal[1] = (float)fbxNormal[1];
      vertex.normal[2] = (float)fbxNormal[2];
      vertex.tangent[0] = (float)fbxTangent[0];
      vertex.tangent[1] = (float)fbxTangent[1];
      vertex.tangent[2] = (float)fbxTangent[2];
      vertex.tangent[3] = (float)fbxTangent[3];
      vertex.binormal[0] = (float)fbxBinormal[0];
      vertex.binormal[1] = (float)fbxBinormal[1];
      vertex.binormal[2] = (float)fbxBinormal[2];
      vertex.color[0] = (float)fbxColor.mRed;
      vertex.color[1] = (float)fbxColor.mGreen;
      vertex.color[2] = (float)fbxColor.mBlue;
      vertex.color[3] = (float)fbxColor.mAlpha;
      vertex.uv0[0] = (float)fbxUV0[0];
      vertex.uv0[1] = (float)fbxUV0[1];
      vertex.uv1[0] = (float)fbxUV1[0];
      vertex.uv1[1] = (float)fbxUV1[1];
      vertex.jointIndices = meshAccess.skinning.GetVertexIndices(controlPointIndex);
      vertex.jointWeights = meshAccess.skinning.GetVertexWeights(controlPointIndex);

      // flag this triangle as transparent if any of its corner vertices substantially deviates
      // from fully opaque
      tri->polygon->hasTranslucentVertices |=
          meshAccess.colorLayer.LayerPresent() && (fabs(fbxColor.mAlpha - 1.0) > 1e-3);

      RawSurface& surface = raw.GetSurface(rawSurfaceIndex);
      surface.bounds.AddPoint(vertex.position);

      if (!triangulation.targetShapes.empty()) {
        vertex.blendSurfaceIx = rawSurfaceIndex;
        for (const auto* targetShape : triangulation.targetShapes) {
          RawBlendVertex blendVertex;
          // the morph target data must be transformed just as with the vertex positions above
          const FbxVector4& shapePosition =
              transform.MultNormalize(targetShape->positions[controlPointIndex]);
          blendVertex.position = toVec3f(shapePosition - fbxPosition) * scaleFactor;
          if (targetShape->normals.LayerPresent()) {
            const FbxVector4& normal = targetShape->normals.GetElement(
                originalPolyIx,
                polygonVertexIndex,
                controlPointIndex,
                FbxVector4(0.0f, 0.0f, 0.0f, 0.0f),
                inverseTransposeTransform,
                true);
            blendVertex.normal = toVec3f(normal - fbxNormal);
          }
          if (targetShape->tangents.LayerPresent()) {
            const FbxVector4& tangent = targetShape->tangents.GetElement(
                originalPolyIx,
                polygonVertexIndex,
                controlPointIndex,
                FbxVector4(0.0f, 0.0f, 0.0f, 0.0f),
                inverseTransposeTransform,
                true);
            blendVertex.tangent = toVec4f(tangent - fbxTangent);
          }
          vertex.blends.push_back(blendVertex);
        }
      } else {
        vertex.blendSurfaceIx = -1;
      }

      if (meshAccess.skinning.IsSkinned()) {
        const int jointIndices[FbxSkinningAccess::MAX_WEIGHTS] = {vertex.jointIndices[0],
                                                                  vertex.jointIndices[1],
                                                                  vertex.jointIndices[2],
                                                                  vertex.jointIndices[3]};
        const float jointWeights[FbxSkinningAccess::MAX_WEIGHTS] = {vertex.jointWeights[0],
                                                                    vertex.jointWeights[1],
                                                                    vertex.jointWeights[2],
                                                                    vertex.jointWeights[3]};
        const FbxMatrix skinningMatrix =
            meshAccess.skinning.GetJointSkinningTransform(jointIndices[0]) * jointWeights[0] +
            meshAccess.skinning.GetJointSkinningTransform(jointIndices[1]) * jointWeights[1] +
            meshAccess.skinning.GetJointSkinningTransform(jointIndices[2]) * jointWeights[2] +
            meshAccess.skinning.GetJointSkinningTransform(jointIndices[3]) * jointWeights[3];

        const FbxVector4 globalPosition = skinningMatrix.MultNormalize(fbxPosition);
        for (int i = 0; i < FbxSkinningAccess::MAX_WEIGHTS; i++) {
          if (jointWeights[i] > 0.0f) {
            const FbxVector4 localPosition =
                meshAccess.skinning.GetJointInverseGlobalTransforms(jointIndices[i])
                    .MultNormalize(globalPosition);

            Vec3f& mins = surface.jointGeometryMins[jointIndices[i]];
            mins[0] = std::min(mins[0], (float)localPosition[0]);
            mins[1] = std::min(mins[1], (float)localPosition[1]);
            mins[2] = std::min(mins[2], (float)localPosition[2]);

            Vec3f& maxs = surface.jointGeometryMaxs[jointIndices[i]];
            maxs[0] = std::max(maxs[0], (float)localPosition[0]);
            maxs[1] = std::max(maxs[1], (float)localPosition[1]);
            maxs[2] = std::max(maxs[2], (float)localPosition[2]);
          }
        }
      }
      int rawVertexIx = raw.AddVertex(rawVertices[vertexIndex]);
      tri->rawVertexIndices[vertexIndex] = rawVertexIx;
    }
  }
}

RawMaterialType FbxMesh2Raw::GetMaterialType(
    const int textures[RAW_TEXTURE_USAGE_MAX],
    const bool vertexTransparency,
    const bool isSkinned) {
  // DIFFUSE and ALBEDO are different enough to represent distinctly, but they both help determine
  // transparency.
  int diffuseTexture = textures[RAW_TEXTURE_USAGE_DIFFUSE];
  if (diffuseTexture < 0) {
    diffuseTexture = textures[RAW_TEXTURE_USAGE_ALBEDO];
  }
  // determine material type based on texture occlusion.
  if (diffuseTexture >= 0) {
    return (raw.GetTexture(diffuseTexture).occlusion == RAW_TEXTURE_OCCLUSION_OPAQUE)
        ? (isSkinned ? RAW_MATERIAL_TYPE_SKINNED_OPAQUE : RAW_MATERIAL_TYPE_OPAQUE)
        : (isSkinned ? RAW_MATERIAL_TYPE_SKINNED_TRANSPARENT : RAW_MATERIAL_TYPE_TRANSPARENT);
  }

  // else if there is any vertex transparency, treat whole mesh as transparent
  if (vertexTransparency) {
    return isSkinned ? RAW_MATERIAL_TYPE_SKINNED_TRANSPARENT : RAW_MATERIAL_TYPE_TRANSPARENT;
  }

  // Default to simply opaque.
  return isSkinned ? RAW_MATERIAL_TYPE_SKINNED_OPAQUE : RAW_MATERIAL_TYPE_OPAQUE;
}

int FbxMesh2Raw::GetMaterial(
    const std::shared_ptr<FbxMaterialInfo> fbxMaterial,
    const std::vector<std::string> userProperties,
    const bool hasTranslucentVertices,
    const bool isSkinned) {
  int textures[RAW_TEXTURE_USAGE_MAX];
  std::fill_n(textures, (int)RAW_TEXTURE_USAGE_MAX, -1);

  std::shared_ptr<RawMatProps> rawMatProps;
  FbxString materialName;
  long materialId;

  if (fbxMaterial == nullptr) {
    materialName = "DefaultMaterial";
    materialId = -1;
    rawMatProps.reset(new RawTraditionalMatProps(
        RAW_SHADING_MODEL_LAMBERT,
        Vec3f(0, 0, 0),
        Vec4f(.5, .5, .5, 1),
        Vec3f(0, 0, 0),
        Vec3f(0, 0, 0),
        0.5));

  } else {
    materialName = fbxMaterial->name;
    materialId = fbxMaterial->id;

    const auto maybeAddTexture = [&](const FbxFileTexture* tex, RawTextureUsage usage) {
      if (tex != nullptr) {
        // dig out the inferred filename from the textureLocations map
        FbxString inferredPath = textureLocations.find(tex)->second;
        textures[usage] =
            raw.AddTexture(tex->GetName(), tex->GetFileName(), inferredPath.Buffer(), usage);
      }
    };

    std::shared_ptr<RawMatProps> matInfo;
    if (fbxMaterial->shadingModel == FbxRoughMetMaterialInfo::FBX_SHADER_METROUGH) {
      FbxRoughMetMaterialInfo* fbxMatInfo =
          static_cast<FbxRoughMetMaterialInfo*>(fbxMaterial.get());

      maybeAddTexture(fbxMatInfo->texBaseColor, RAW_TEXTURE_USAGE_ALBEDO);
      maybeAddTexture(fbxMatInfo->texNormal, RAW_TEXTURE_USAGE_NORMAL);
      maybeAddTexture(fbxMatInfo->texEmissive, RAW_TEXTURE_USAGE_EMISSIVE);
      maybeAddTexture(fbxMatInfo->texRoughness, RAW_TEXTURE_USAGE_ROUGHNESS);
      maybeAddTexture(fbxMatInfo->texMetallic, RAW_TEXTURE_USAGE_METALLIC);
      maybeAddTexture(fbxMatInfo->texAmbientOcclusion, RAW_TEXTURE_USAGE_OCCLUSION);
      rawMatProps.reset(new RawMetRoughMatProps(
          RAW_SHADING_MODEL_PBR_MET_ROUGH,
          toVec4f(fbxMatInfo->baseColor),
          toVec3f(fbxMatInfo->emissive),
          fbxMatInfo->emissiveIntensity,
          fbxMatInfo->metallic,
          fbxMatInfo->roughness,
          fbxMatInfo->invertRoughnessMap));
    } else {
      FbxTraditionalMaterialInfo* fbxMatInfo =
          static_cast<FbxTraditionalMaterialInfo*>(fbxMaterial.get());
      RawShadingModel shadingModel;
      if (fbxMaterial->shadingModel == "Lambert") {
        shadingModel = RAW_SHADING_MODEL_LAMBERT;
      } else if (0 == fbxMaterial->shadingModel.CompareNoCase("Blinn")) {
        shadingModel = RAW_SHADING_MODEL_BLINN;
      } else if (0 == fbxMaterial->shadingModel.CompareNoCase("Phong")) {
        shadingModel = RAW_SHADING_MODEL_PHONG;
      } else if (0 == fbxMaterial->shadingModel.CompareNoCase("Constant")) {
        shadingModel = RAW_SHADING_MODEL_PHONG;
      } else {
        shadingModel = RAW_SHADING_MODEL_UNKNOWN;
      }
      maybeAddTexture(fbxMatInfo->texDiffuse, RAW_TEXTURE_USAGE_DIFFUSE);
      maybeAddTexture(fbxMatInfo->texNormal, RAW_TEXTURE_USAGE_NORMAL);
      maybeAddTexture(fbxMatInfo->texEmissive, RAW_TEXTURE_USAGE_EMISSIVE);
      maybeAddTexture(fbxMatInfo->texShininess, RAW_TEXTURE_USAGE_SHININESS);
      maybeAddTexture(fbxMatInfo->texAmbient, RAW_TEXTURE_USAGE_AMBIENT);
      maybeAddTexture(fbxMatInfo->texSpecular, RAW_TEXTURE_USAGE_SPECULAR);
      rawMatProps.reset(new RawTraditionalMatProps(
          shadingModel,
          toVec3f(fbxMatInfo->colAmbient),
          toVec4f(fbxMatInfo->colDiffuse),
          toVec3f(fbxMatInfo->colEmissive),
          toVec3f(fbxMatInfo->colSpecular),
          fbxMatInfo->shininess));
    }
  }

  return raw.AddMaterial(
      materialId,
      materialName,
      GetMaterialType(textures, hasTranslucentVertices, isSkinned),
      textures,
      rawMatProps,
      userProperties);
}

void FbxMesh2Raw::ProcessPolygons(
    Triangulation& triangulation,
    int rawSurfaceIndex,
    const FbxMeshAccess& meshAccess) {
  for (auto& poly : triangulation.polygons) {
    assert(poly->rawPolyIndex == -1);
    int rawMaterialIx = GetMaterial(
        meshAccess.materials.GetMaterial(poly->fbxPolyIndex),
        meshAccess.materials.GetUserProperties(poly->fbxPolyIndex),
        poly->hasTranslucentVertices,
        meshAccess.skinning.IsSkinned());
    int rawPolyIx = raw.AddPolygon(rawSurfaceIndex, rawMaterialIx);
    poly->rawPolyIndex = rawPolyIx;
  }
  // now we have everything we need to actually create the triangles
  for (const auto& tri : triangulation.triangles) {
    assert(tri->polygon->rawPolyIndex >= 0);
    raw.AddTriangle(
        tri->rawVertexIndices[0],
        tri->rawVertexIndices[1],
        tri->rawVertexIndices[2],
        tri->polygon->rawPolyIndex);
  }
}

std::unique_ptr<Triangulation> FbxMesh2Raw::TriangulateUsingSDK() {
  FbxGeometryConverter meshConverter(scene->GetFbxManager());
  meshConverter.Triangulate(node->GetNodeAttribute(), true);
  FbxMesh* mesh = node->GetMesh();
  auto result = std::make_unique<Triangulation>(mesh);

  int polygonVertexIndex = 0;
  for (int fbxPolyIx = 0; fbxPolyIx < mesh->GetPolygonCount(); fbxPolyIx++) {
    FBX_ASSERT(mesh->GetPolygonSize(fbxPolyIx) == 3);

    auto polygon = std::make_shared<Polygon>(fbxPolyIx);
    auto triangle = std::make_shared<Triangle>(
        polygon, polygonVertexIndex + 0, polygonVertexIndex + 1, polygonVertexIndex + 2);
    result->polygons.push_back(polygon);
    result->triangles.push_back(triangle);
    polygonVertexIndex += 3;
  }
  return result;
}

std::unique_ptr<Triangulation> FbxMesh2Raw::TriangulateForNgonEncoding() {
  assert(node->GetNodeAttribute()->GetAttributeType() == FbxNodeAttribute::eMesh);
  auto result = std::make_unique<Triangulation>(mesh);
  FbxMesh* mesh = node->GetMesh();

  int* allPolygonVertices = mesh->GetPolygonVertices();
  int anchorIndex = -1;
  for (int polygonIndex = 0; polygonIndex < mesh->GetPolygonCount(); polygonIndex++) {
    int polyVertexCount = mesh->GetPolygonSize(polygonIndex);
    int polyVertexStart = mesh->GetPolygonVertexIndex(polygonIndex);
    int offset = (anchorIndex == allPolygonVertices[polyVertexCount]) ? 1 : 0;
    anchorIndex = allPolygonVertices[polyVertexStart + offset];

    auto polygon = std::make_shared<Polygon>(polygonIndex);
    result->polygons.push_back(polygon);
    for (int vertexIndex = offset + 2; vertexIndex < offset + polyVertexCount; vertexIndex++) {
      int firstTriIndex = vertexIndex - 1;
      int otherTriIndex = vertexIndex % polyVertexCount;

      result->triangles.push_back(std::make_shared<Triangle>(
          polygon,
          polyVertexStart + offset,
          polyVertexStart + firstTriIndex,
          polyVertexStart + otherTriIndex));
    }
  }
  return result;
}
