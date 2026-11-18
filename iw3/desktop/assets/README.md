# 自然夜景素材来源

`lakeside_night.png` 是 Poly Haven 正式 HDRI **Lakeside Night** 的离线转换版本。

- 原作者：Greg Zaal（Photography），Jarod Guest（Processing）。
- 官方资产页：https://polyhaven.com/a/lakeside_night
- 官方资产许可：**CC0 1.0**，https://polyhaven.com/license ，https://creativecommons.org/publicdomain/zero/1.0/ 。允许修改、商业使用和再分发。
- 本次正式源文件：https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/lakeside_night_2k.hdr
- 源文件大小：6,740,809 bytes；MD5：`21b610f35a5c5629dacdec4e44bcca84`。
- 取得与转换日期：2026-11-18（Asia/Shanghai）。

使用原因：用户选择“自然夜景＋视频柔光”，用真实星空、湖面和岸线提供稳定空间参照。主视频保持观看重点，夜景底图保留自然色，视频方向柔光单独更新。

转换步骤：OpenCV读取正式2K Radiance HDR（实测float32 BGR 2048×1024），显式转RGB，固定曝光0.30、按亮度压缩高光、gamma2.2、最高38%显示白。随后按现有OpenXR六面方向投影为每面512×512，四边复制2px护边，拼成3×2、1548×1032 RGBA PNG。运行时仅PIL加载PNG，不联网、不依赖OpenCV、不读取HDR，不随柔光亮度反复发送高清底图。

原HDR与离线转换脚本位于项目研究目录，不作为应用资源或用户媒体。这里使用正式CC0 HDRI文件的衍生图，未使用网站宣传图、网页预览截图、Logo或第三方示例渲染。
