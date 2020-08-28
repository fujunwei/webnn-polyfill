import { Output } from './Output';
import { Input } from './Input';
import { Constant } from './Constant';
import { Operation } from './Operation';
import { CompilationOptions } from './CompilationOptions';
import { Compilation } from './Compilation';
import { NamedOperand } from './NamedOperand';
import * as utils from './utils';

/**
 * Implements the [Model](https://webmachinelearning.github.io/webnn/#model) interface.
 */
export class Model {
  private inputs_: Map<string, Input> = new Map();
  private outputs_: Map<string, Output> = new Map();
  private constants_: Constant[] = [];

  get inputs() { return this.inputs_; }
  get outputs() { return this.outputs_; }
  get constants() { return this.constants_; }

  constructor(outputs: NamedOperand[]) {
    utils.assert(outputs.length !== 0, 'The length of outputs parameter should not be 0.');
    utils.assert(outputs.every(namedOutput => typeof namedOutput.name === 'string' &&
        namedOutput.operand instanceof Output), 'The outputs parameter is invalid.');
    for (const namedOutput of outputs) {
      this.outputs_.set(namedOutput.name, namedOutput.operand as Output);
    }
    this.initialize_();
  }

  /** */
  async createCompilation(options: CompilationOptions): Promise<Compilation> {
    const compilation = await Compilation.createAndCompile(options, this);
    return compilation;
  }

  private initialize_(): void {
    const self = this;
    function handleOperation(operation: Operation): void {
      for (const operand of operation.inputs) {
        if (operand instanceof Input) {
          self.inputs_.set(operand.name, operand);
        } else if (operand instanceof Constant) {
          self.constants_.push(operand);
        } else if (operand instanceof Output) {
          handleOperation(operand.operation);
        }
      }
    }
    for (const output of this.outputs_.values()) {
      handleOperation(output.operation);
    }
  }
}