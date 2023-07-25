# BarcodeBlemish
针对一维和二维条码的缺陷检测模型

</br>

# 模型说明
<table>
    <tr>
	    <th>模型</th>
	    <th>说明</th>
	    <th>数据集</th> 
        <th>输入（n,c,h,w）</th>  
        <th>输出</th>   
	</tr >
    <tr >
	    <td></td>
        <td></td>
        <td></td>
        <td></td>
	    <td>0：条码区域 1：缺陷区域 2：背景区域</td>
	</tr>
	<tr >
	    <td>1D.v1.onnx</td>
	    <td>一维码缺陷检测</td>
	    <td>code39+code128</td>
        <td>(1,1,384,960)</td>
        <td>同输入一致</td>
	</tr>
	<tr >
	    <td>1D.v2.onnx</td>
	    <td>一维码缺陷检测</td>
	    <td>code39+code128</td>
        <td>(1,1,384,960)</td>
        <td>同输入一致</td>
	</tr>
	<tr>
	    <td></td>
	    <td></td>
	    <td></td>
        <td>(1,1,192,490)</td>
        <td></td>
	</tr>
	<tr>
	    <td>2D.square.v1.onnxdio</td>
	    <td>二维码缺陷检测，主要针对于方块码</td>
	    <td>qr+datamatrix</td>
	    <td>1,1,480,480 </td>
	    <td>同输入一致</td>
	</tr>
	<tr>
	    <td></td>
	    <td></td>
	    <td></td>
        <td>支持动态大小</td>
        <td></td>
	</tr>
	<tr>
	    <td>2D.square.v2.onnxdio</td>
	    <td>二维码缺陷检测，主要针对于方块码</td>
	    <td>qr+datamatrix</td>
	    <td>1,1,480,480 </td>
	    <td>同输入一致</td>
	</tr>
	<tr>
	    <td></td>
	    <td></td>
	    <td></td>
        <td>支持动态大小</td>
        <td></td>
	</tr>
</table>