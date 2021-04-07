'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const context = navigator.ml.createContext();

  it('test clamp converted from relu1_float_1 test', async function() {
    // Converted test case (from: V1_0/relu1_float_1.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const op1Data = new Float32Array([-10.0, -0.5, 0.5, 10.0]);
    const expected = [-1.0, -0.5, 0.5, 1.0];
    const op2 = builder.clamp(op1, {minValue: builder.constant(-1), maxValue: builder.constant(1)});
    const graph = await builder.build({op2});
    const outputs = await graph.compute({'op1': {data: op1Data}});
    utils.checkValue(outputs.op2.data, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
