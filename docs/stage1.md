### 关系全集
Note: 涉及Polygon和Ellipse的相切关系时，relation统一为[Ellipse_idx, Polygon_idx, relation_type]，即顺序为椭圆内切多边形/椭圆外接多边形。

1. 基于Polygon生成的关系
- tangent line: 有一条线与某个顶点相切
- symmetric: 有另一个多边形关于某条边对称
- similar: 和另一个多边形相似
- shared edge: 和另一个多边形共用一条边，仅出现在三角形中
- circumscribed circle of triangle: 三角形有一个外接圆
- circumscribed circle of rectangle: 矩形有一个外接圆
- inscribed: 有一个内切圆，仅出现在三角形中
- diagonal: 有一个对角线，仅出现在矩形中

2. 基于Line生成的关系
- parallel: 有一条平行线
- tangent line: 有一个椭圆或多边形，与Line相切
- major axis: Line是一个椭圆的长轴
- minor axis: Line是一个椭圆的短轴
- diamaeter: Line是一个圆的直径

3. 基于Ellipse生成的关系
- tangent line: 有一条切线
- internal tangent circle: 有一个与椭圆内切的圆
- external tangent circle: 有一个与椭圆外切的圆
- concentric: 有一组同心椭圆
- circumscribed: 椭圆外接于一个多边形
- inscribed: 椭圆内切于一个多边形

### special_info全集
1. Polygon
- triangle: 三角形
- equilateral triangle: 等边三角形
- rectangle: 矩形
2. Ellipse
- circle: 圆